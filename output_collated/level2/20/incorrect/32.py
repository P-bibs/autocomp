# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_130922/code_23.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'output_padding', 'bias_shape']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'bias']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a 3D transposed convolution, followed by a sum, 
    a residual add, a multiplication, and another residual add.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, *args) -> torch.Tensor:
        return functional_model(*args, **extract_state_kwargs(self))


def build_reference_model():
    init_inputs = list(get_init_inputs())
    model = ModelNew(*init_inputs)
    model.eval()
    return model


def extract_state_kwargs(model):
    flat_state = {}
    for name, value in model.named_parameters():
        flat_state[name.replace('.', '_')] = value
    for name, value in model.named_buffers():
        flat_state[name.replace('.', '_')] = value
    state_kwargs = {}
    init_inputs = list(get_init_inputs())
    init_arg_map = {name: value for name, value in zip(INIT_PARAM_NAMES, init_inputs)}
    # State for conv_transpose (nn.ConvTranspose3d)
    if 'conv_transpose_weight' in flat_state:
        state_kwargs['conv_transpose_weight'] = flat_state['conv_transpose_weight']
    else:
        state_kwargs['conv_transpose_weight'] = getattr(model.conv_transpose, 'weight', None)
    if 'conv_transpose_bias' in flat_state:
        state_kwargs['conv_transpose_bias'] = flat_state['conv_transpose_bias']
    else:
        state_kwargs['conv_transpose_bias'] = getattr(model.conv_transpose, 'bias', None)
    state_kwargs['conv_transpose_stride'] = model.conv_transpose.stride
    state_kwargs['conv_transpose_padding'] = model.conv_transpose.padding
    state_kwargs['conv_transpose_output_padding'] = model.conv_transpose.output_padding
    state_kwargs['conv_transpose_groups'] = model.conv_transpose.groups
    state_kwargs['conv_transpose_dilation'] = model.conv_transpose.dilation
    if 'bias' in flat_state:
        state_kwargs['bias'] = flat_state['bias']
    else:
        state_kwargs['bias'] = getattr(model, 'bias')
    missing = [name for name in REQUIRED_STATE_NAMES if name not in state_kwargs]
    if missing:
        raise RuntimeError(f'Missing required state entries: {missing}')
    return state_kwargs


def get_functional_inputs():
    model = build_reference_model()
    forward_args = tuple(get_inputs())
    state_kwargs = extract_state_kwargs(model)
    return forward_args, state_kwargs




import torch
from torch.utils.cpp_extension import load_inline

# Manual 3D Transpose Conv + Fused Arithmetic
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// Direct 3D Transposed Convolution Kernel (Simplified for standard strides)
// Projects input channels to output via weight kernel
__global__ void conv_transpose3d_fused_kernel(
    const float* __restrict__ in,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int N, int IC, int ID, int IH, int IW,
    int OC, int OD, int OH, int OW,
    int KD, int KH, int KW,
    int stride, int pad
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= N * OC * OD * OH * OW) return;

    int tmp = out_idx;
    int w = tmp % OW; tmp /= OW;
    int h = tmp % OH; tmp /= OH;
    int d = tmp % OD; tmp /= OD;
    int oc = tmp % OC; tmp /= OC;
    int n = tmp;

    float acc = 0.0f;
    // Map output coordinate to input space
    for (int ic = 0; ic < IC; ++ic) {
        for (int kd = 0; kd < KD; ++kd) {
            int id = (d + pad - kd);
            if (id % stride == 0) {
                id /= stride;
                for (int kh = 0; kh < KH; ++kh) {
                    int ih = (h + pad - kh);
                    if (ih % stride == 0) {
                        ih /= stride;
                        for (int kw = 0; kw < KW; ++kw) {
                            int iw = (w + pad - kw);
                            if (iw % stride == 0) {
                                iw /= stride;
                                if (id >= 0 && id < ID && ih >= 0 && ih < IH && iw >= 0 && iw < IW) {
                                    float val = in[((n * IC + ic) * ID + id) * IH * IW + ih * IW + iw];
                                    float w_val = weight[(((ic * OC + oc) * KD + kd) * KH + kh) * KW + kw];
                                    acc += val * w_val;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    float b_val = bias[oc];
    // Fused Arithmetic: (2*x^2 + x*b + x)
    out[out_idx] = (2.0f * acc * acc) + (acc * b_val) + acc;
}

void launch_fused_conv(
    torch::Tensor in, torch::Tensor weight, torch::Tensor bias, torch::Tensor out,
    int stride, int pad
) {
    int N = in.size(0); int IC = in.size(1); int ID = in.size(2); int IH = in.size(3); int IW = in.size(4);
    int OC = out.size(1); int OD = out.size(2); int OH = out.size(3); int OW = out.size(4);
    int KD = weight.size(2); int KH = weight.size(3); int KW = weight.size(4);

    int total_threads = N * OC * OD * OH * OW;
    int threads = 256;
    int blocks = (total_threads + threads - 1) / threads;

    conv_transpose3d_fused_kernel<<<blocks, threads>>>(
        in.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), out.data_ptr<float>(),
        N, IC, ID, IH, IW, OC, OD, OH, OW, KD, KH, KW, stride, pad
    );
}
"""

cpp_source = r"""
void launch_fused_conv(torch::Tensor in, torch::Tensor weight, torch::Tensor bias, torch::Tensor out, int stride, int pad);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_transpose", &launch_fused_conv, "Fused Transpose Conv and Ops");
}
"""

module = load_inline(
    name='fused_conv_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, conv_transpose_weight, conv_transpose_bias, 
                     conv_transpose_stride, conv_transpose_padding, 
                     conv_transpose_output_padding, conv_transpose_groups, 
                     conv_transpose_dilation, bias):
    # Output dims calculation
    N, IC, ID, IH, IW = x.shape
    OC, KD, KH, KW = conv_transpose_weight.shape[1], *conv_transpose_weight.shape[2:]
    out = torch.empty((N, OC, (ID-1)*conv_transpose_stride + KD - 2*conv_transpose_padding, 
                       (IH-1)*conv_transpose_stride + KH - 2*conv_transpose_padding, 
                       (IW-1)*conv_transpose_stride + KW - 2*conv_transpose_padding), device='cuda')
    
    module.fused_conv_transpose(x, conv_transpose_weight, bias.view(-1), out, 
                                conv_transpose_stride, conv_transpose_padding)
    return out
