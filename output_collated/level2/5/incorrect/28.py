# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_114641/code_15.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'bias_shape', 'stride', 'padding', 'output_padding']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'bias']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a transposed convolution, subtracts a bias term, and applies tanh activation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, bias_shape, stride=2, padding=1, output_padding=1):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
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
    # State for conv_transpose (nn.ConvTranspose2d)
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

# 1. CUDA Source: Fused kernel performing Transposed Conv followed by element-wise operations
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_transpose_conv_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ sub_bias,
    float* __restrict__ output,
    const int N, const int C_in, const int C_out,
    const int H_in, const int W_in,
    const int H_out, const int W_out,
    const int stride, const int padding,
    const int dilation, const int kernel_size)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * C_out * H_out * W_out;
    if (idx >= total) return;

    int temp = idx;
    const int wo = temp % W_out; temp /= W_out;
    const int ho = temp % H_out; temp /= H_out;
    const int co = temp % C_out; temp /= C_out;
    const int n  = temp;

    float sum = 0.0f;

    for (int ci = 0; ci < C_in; ++ci) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            int hi_unscaled = ho + padding - kh * dilation;
            if (hi_unscaled % stride != 0) continue;
            int hi = hi_unscaled / stride;
            if (hi < 0 || hi >= H_in) continue;

            for (int kw = 0; kw < kernel_size; ++kw) {
                int wi_unscaled = wo + padding - kw * dilation;
                if (wi_unscaled % stride != 0) continue;
                int wi = wi_unscaled / stride;
                if (wi < 0 || wi >= W_in) continue;

                float val = __ldg(&input[((n * C_in + ci) * H_in + hi) * W_in + wi]);
                float w = __ldg(&weight[((ci * C_out + co) * kernel_size + kh) * kernel_size + kw]);
                sum += val * w;
            }
        }
    }

    sum += __ldg(&conv_bias[co]);
    sum -= __ldg(&sub_bias[co]);
    output[idx] = tanhf(sum);
}

void fused_op_forward(
    torch::Tensor input, torch::Tensor weight, torch::Tensor conv_bias,
    torch::Tensor sub_bias, torch::Tensor output,
    int N, int C_in, int C_out, int H_in, int W_in,
    int H_out, int W_out, int stride, int padding,
    int dilation, int kernel_size)
{
    const int total = N * C_out * H_out * W_out;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    fused_transpose_conv_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), 
        conv_bias.data_ptr<float>(), sub_bias.data_ptr<float>(),
        output.data_ptr<float>(), N, C_in, C_out, H_in, W_in, 
        H_out, W_out, stride, padding, dilation, kernel_size
    );
}
"""

cpp_source = r"""
void fused_op_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor conv_bias,
                      torch::Tensor sub_bias, torch::Tensor output,
                      int N, int C_in, int C_out, int H_in, int W_in,
                      int H_out, int W_out, int stride, int padding,
                      int dilation, int kernel_size);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused Transposed Conv");
}
"""

fused_ext = load_inline(name='fused_op', cpp_sources=cpp_source, cuda_sources=cuda_source,
                        extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True)

def functional_model(x, *, conv_transpose_weight, conv_transpose_bias, conv_transpose_stride,
                     conv_transpose_padding, conv_transpose_output_padding, conv_transpose_groups,
                     conv_transpose_dilation, bias):
    N, C_in, H_in, W_in = x.shape
    C_out = conv_transpose_weight.size(1)
    K = conv_transpose_weight.size(2)
    H_out = (H_in - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_dilation * (K - 1) + conv_transpose_output_padding + 1
    W_out = (W_in - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_dilation * (K - 1) + conv_transpose_output_padding + 1

    output = torch.empty((N, C_out, H_out, W_out), device=x.device, dtype=x.dtype)
    fused_ext.fused_op(x.contiguous(), conv_transpose_weight.contiguous(), 
                       conv_transpose_bias.contiguous(), bias.contiguous().view(-1), 
                       output, N, C_in, C_out, H_in, W_in, H_out, W_out, 
                       conv_transpose_stride, conv_transpose_padding, 
                       conv_transpose_dilation, K)
    return output
