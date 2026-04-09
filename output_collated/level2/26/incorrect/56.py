# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_045050/code_19.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'output_padding', 'bias_shape']
FORWARD_ARG_NAMES = ['x', 'add_input']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'bias']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a 3D transposed convolution, adds an input tensor, and applies HardSwish activation.
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

# The CUDA kernel performs a direct mapping (im2col style logic integrated)
# of the transpose convolution: For each output pixel, determine participating 
# input pixels that contribute to it (inverse of conv3d logic).
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_conv_tr_add_hs_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ add_input,
    float* __restrict__ out,
    int N, int in_channels, int out_channels,
    int D, int H, int W, int OD, int OH, int OW,
    int KD, int KH, int KW, int stride, int padding) 
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    // Map linear index to output coordinates
    int ow = tid % OW;
    int oh = (tid / OW) % OH;
    int od = (tid / (OW * OH)) % OD;
    int oc = (tid / (OW * OH * OD)) % out_channels;
    int b  = tid / (OW * OH * OD * out_channels);

    float val = bias[oc];

    // Transpose Conv accumulation
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kd = 0; kd < KD; ++kd) {
            int id = (od + padding - kd);
            if (id % stride != 0) continue;
            id /= stride;
            if (id < 0 || id >= D) continue;

            for (int kh = 0; kh < KH; ++kh) {
                int ih = (oh + padding - kh);
                if (ih % stride != 0) continue;
                ih /= stride;
                if (ih < 0 || ih >= H) continue;

                for (int kw = 0; kw < KW; ++kw) {
                    int iw = (ow + padding - kw);
                    if (iw % stride != 0) continue;
                    iw /= stride;
                    if (iw < 0 || iw >= W) continue;

                    int w_idx = (((oc * in_channels + ic) * KD + kd) * KH + kh) * KW + kw;
                    int i_idx = (((b * in_channels + ic) * D + id) * H + ih) * W + iw;
                    val += input[i_idx] * weight[w_idx];
                }
            }
        }
    }

    // Fused Add + HardSwish
    val += add_input[tid];
    float hswish = val * fminf(fmaxf(val + 3.0f, 0.0f), 6.0f) * 0.16666666666666666f;
    out[tid] = val * hswish;
}

void fused_op_forward(torch::Tensor in, torch::Tensor w, torch::Tensor b, torch::Tensor add, torch::Tensor out,
                      int D, int H, int W, int OD, int OH, int OW,
                      int KD, int KH, int KW, int stride, int padding) {
    int N = out.numel();
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    fused_conv_tr_add_hs_kernel<<<blocks, threads>>>(
        in.data_ptr<float>(), w.data_ptr<float>(), b.data_ptr<float>(), 
        add.data_ptr<float>(), out.data_ptr<float>(),
        N, in.size(1), out.size(1), D, H, W, OD, OH, OW, KD, KH, KW, stride, padding
    );
}
"""

cpp_source = r"""
void fused_op_forward(torch::Tensor in, torch::Tensor w, torch::Tensor b, torch::Tensor add, torch::Tensor out,
                      int D, int H, int W, int OD, int OH, int OW,
                      int KD, int KH, int KW, int stride, int padding);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused Transpose Conv3D + Add + HardSwish");
}
"""

fused_ext = load_inline(name='fused_op', cpp_sources=cpp_source, cuda_sources=cuda_source, 
                        extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True)

def functional_model(x, add_input, *, conv_transpose_weight, conv_transpose_bias, conv_transpose_stride,
                     conv_transpose_padding, conv_transpose_output_padding, conv_transpose_groups, 
                     conv_transpose_dilation, bias):
    # Determine output shape
    batch, in_c, D, H, W = x.shape
    out_c, _, KD, KH, KW = conv_transpose_weight.shape
    OD = (D - 1) * conv_transpose_stride - 2 * conv_transpose_padding + KD + conv_transpose_output_padding
    OH = (H - 1) * conv_transpose_stride - 2 * conv_transpose_padding + KH + conv_transpose_output_padding
    OW = (W - 1) * conv_transpose_stride - 2 * conv_transpose_padding + KW + conv_transpose_output_padding
    
    out = torch.empty((batch, out_c, OD, OH, OW), device=x.device)
    fused_ext.fused_op(x, conv_transpose_weight, conv_transpose_bias, add_input, out,
                       D, H, W, OD, OH, OW, KD, KH, KW, conv_transpose_stride, conv_transpose_padding)
    return out
