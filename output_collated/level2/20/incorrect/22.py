# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_130013/code_21.py
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

# -------------------------------------------------------------------------
# Fused Transposed Conv3d + Arithmetic Kernel
# Optimized to minimize global accesses by computing post-arithmetic 
# during the output write stage.
# -------------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_conv_transpose3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int IC, int OC, int ID, int IH, int IW,
    int KD, int KH, int KW, int OD, int OH, int OW,
    int stride, int padding) 
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N * OC * OD * OH * OW) return;

    // Map linear ID back to [N, OC, d, h, w]
    int w = tid % OW;
    int h = (tid / OW) % OH;
    int d = (tid / (OW * OH)) % OD;
    int oc = (tid / (OW * OH * OD)) % OC;
    int n = tid / (OW * OH * OD * OC);

    float acc = 0.0f;
    // Standard naive Transpose Conv (optimized with register accumulation)
    for (int ic = 0; ic < IC; ++ic) {
        for (int kd = 0; kd < KD; ++kd) {
            for (int kh = 0; kh < KH; ++kh) {
                for (int kw = 0; kw < KW; ++kw) {
                    int id_in = (d + padding - kd) / stride;
                    int ih_in = (h + padding - kh) / stride;
                    int iw_in = (w + padding - kw) / stride;
                    
                    if ((d + padding - kd) % stride == 0 &&
                        (h + padding - kh) % stride == 0 &&
                        (w + padding - kw) % stride == 0 &&
                        id_in >= 0 && id_in < ID &&
                        ih_in >= 0 && ih_in < IH &&
                        iw_in >= 0 && iw_in < IW) {
                        
                        float x = input[((n * IC + ic) * ID + id_in) * IH * IW + ih_in * IW + iw_in];
                        float w_val = weight[(((ic * OC + oc) * KD + kd) * KH + kh) * KW + kw];
                        acc += x * w_val;
                    }
                }
            }
        }
    }

    // Fused post-arithmetic: 2*x^2 + b*x + x
    float b_val = bias[oc];
    float res = (2.0f * acc * acc) + (b_val * acc) + acc;
    output[tid] = res;
}

void launch_fused_conv(const torch::Tensor& in, const torch::Tensor& weight, const torch::Tensor& bias, torch::Tensor& out, int s, int p) {
    int N = in.size(0); int IC = in.size(1); int ID = in.size(2); int IH = in.size(3); int IW = in.size(4);
    int OC = weight.size(1);
    int KD = weight.size(2); int KH = weight.size(3); int KW = weight.size(4);
    int OD = out.size(2); int OH = out.size(3); int OW = out.size(4);
    
    int num_elements = N * OC * OD * OH * OW;
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;
    
    fused_conv_transpose3d_kernel<<<blocks, threads>>>(
        in.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), out.data_ptr<float>(),
        N, IC, OC, ID, IH, IW, KD, KH, KW, OD, OH, OW, s, p
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void launch_fused_conv(const torch::Tensor& in, const torch::Tensor& weight, const torch::Tensor& bias, torch::Tensor& out, int s, int p);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv", &launch_fused_conv, "Fused Transposed Conv + Arithmetic");
}
"""

fused_ext = load_inline(name='fused_conv_cuda', cpp_sources=cpp_source, cuda_sources=cuda_kernel, extra_cuda_cflags=['-O3'], with_cuda=True)

def functional_model(x, *, conv_transpose_weight, conv_transpose_bias, conv_transpose_stride, 
                     conv_transpose_padding, conv_transpose_output_padding, conv_transpose_groups, 
                     conv_transpose_dilation, bias):
    # Calculate output shape
    N, IC, D, H, W = x.shape
    OC, _, KD, KH, KW = conv_transpose_weight.shape
    OD = (D - 1) * conv_transpose_stride - 2 * conv_transpose_padding + KD + conv_transpose_output_padding
    OH = (H - 1) * conv_transpose_stride - 2 * conv_transpose_padding + KH + conv_transpose_output_padding
    OW = (W - 1) * conv_transpose_stride - 2 * conv_transpose_padding + KW + conv_transpose_output_padding
    
    output = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)
    fused_ext.fused_conv(x, conv_transpose_weight, bias, output, conv_transpose_stride, conv_transpose_padding)
    return output
