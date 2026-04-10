# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_123708/code_7.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'min_value', 'divisor']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'min_value', 'divisor']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias']


class ModelNew(nn.Module):
    """
    A model that performs a transposed 3D convolution, clamps the output to a minimum value, 
    and then divides the result by a constant.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, min_value, divisor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.min_value = min_value
        self.divisor = divisor

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
    if 'min_value' in flat_state:
        state_kwargs['min_value'] = flat_state['min_value']
    else:
        state_kwargs['min_value'] = getattr(model, 'min_value')
    if 'divisor' in flat_state:
        state_kwargs['divisor'] = flat_state['divisor']
    else:
        state_kwargs['divisor'] = getattr(model, 'divisor')
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

# ----------------------------------------------------------------------
# CUDA source – Fused Transposed Conv3D + Clamp + Division
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_transpose_conv_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int N, const int C_in, const int C_out,
    const int D_in, const int H_in, const int W_in,
    const int D_out, const int H_out, const int W_out,
    const int K, const int stride, const int padding,
    const int dilation, const float min_val, const float divisor)
{
    const int total_out = N * C_out * D_out * H_out * W_out;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_out; idx += blockDim.x * gridDim.x) {
        int tmp = idx;
        const int w = tmp % W_out; tmp /= W_out;
        const int h = tmp % H_out; tmp /= H_out;
        const int d = tmp % D_out; tmp /= D_out;
        const int co = tmp % C_out; tmp /= C_out;
        const int n = tmp;

        float acc = bias[co];

        // Transposed convolution math: iterate input patches that contribute to current output pixel
        for (int ci = 0; ci < C_in; ++ci) {
            const int w_off = ((co * C_in + ci) * K * K * K);
            
            #pragma unroll
            for (int kd = 0; kd < K; ++kd) {
                int d_in_idx = d + padding - kd * dilation;
                if (d_in_idx < 0 || d_in_idx % stride != 0) continue;
                int in_d = d_in_idx / stride;
                if (in_d >= D_in) continue;

                #pragma unroll
                for (int kh = 0; kh < K; ++kh) {
                    int h_in_idx = h + padding - kh * dilation;
                    if (h_in_idx < 0 || h_in_idx % stride != 0) continue;
                    int in_h = h_in_idx / stride;
                    if (in_h >= H_in) continue;

                    #pragma unroll
                    for (int kw = 0; kw < K; ++kw) {
                        int w_in_idx = w + padding - kw * dilation;
                        if (w_in_idx < 0 || w_in_idx % stride != 0) continue;
                        int in_w = w_in_idx / stride;
                        if (in_w >= W_in) continue;

                        int iIdx = (((n * C_in + ci) * D_in + in_d) * H_in + in_h) * W_in + in_w;
                        int wIdx = w_off + ((kd * K + kh) * K + kw);
                        acc += input[iIdx] * weight[wIdx];
                    }
                }
            }
        }
        output[idx] = fmaxf(acc, min_val) / divisor;
    }
}

void fused_transpose_conv(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output,
    int N, int C_in, int C_out, int D_in, int H_in, int W_in,
    int D_out, int H_out, int W_out, int K, int stride, 
    int padding, int dilation, float min_val, float divisor)
{
    const int total_ele = N * C_out * D_out * H_out * W_out;
    const int threads = 256;
    const int blocks = (total_ele + threads - 1) / threads;
    fused_transpose_conv_kernel<<<min(blocks, 65535), threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), 
        output.data_ptr<float>(), N, C_in, C_out, D_in, H_in, W_in, 
        D_out, H_out, W_out, K, stride, padding, dilation, min_val, divisor);
}
"""

cpp_source = r"""
void fused_transpose_conv(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output,
                          int N, int C_in, int C_out, int D_in, int H_in, int W_in, int D_out, int H_out, int W_out,
                          int K, int stride, int padding, int dilation, float min_val, float divisor);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_transpose_conv", &fused_transpose_conv, "Fused Transposed Conv3D");
}
"""

fused_ext = load_inline(
    name='fused_transpose_op', cpp_sources=cpp_source, cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True
)

def functional_model(x, *, conv_transpose_weight, conv_transpose_bias, conv_transpose_stride,
                     conv_transpose_padding, conv_transpose_output_padding, conv_transpose_groups,
                     conv_transpose_dilation, min_value, divisor):
    x = x.contiguous().cuda()
    w = conv_transpose_weight.contiguous().cuda()
    b = conv_transpose_bias.contiguous().cuda()
    
    N, C_in, D_in, H_in, W_in = x.shape
    C_out, _, K, _, _ = w.shape
    
    D_out = (D_in - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_dilation * (K - 1) + conv_transpose_output_padding + 1
    H_out = (H_in - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_dilation * (K - 1) + conv_transpose_output_padding + 1
    W_out = (W_in - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_dilation * (K - 1) + conv_transpose_output_padding + 1
    
    out = torch.empty((N, C_out, D_out, H_out, W_out), dtype=x.dtype, device=x.device)
    fused_ext.fused_transpose_conv(x, w, b, out, N, C_in, C_out, D_in, H_in, W_in, D_out, H_out, W_out, 
                                   K, conv_transpose_stride, conv_transpose_padding, conv_transpose_dilation, min_value, divisor)
    return out
