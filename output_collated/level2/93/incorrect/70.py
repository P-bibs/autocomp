# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_160727/code_15.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'add_value', 'multiply_value']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'add_value', 'multiply_value']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a transposed convolution, adds a value, takes the minimum, applies GELU, and multiplies by a value.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, add_value, multiply_value):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.add_value = add_value
        self.multiply_value = multiply_value

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
    if 'add_value' in flat_state:
        state_kwargs['add_value'] = flat_state['add_value']
    else:
        state_kwargs['add_value'] = getattr(model, 'add_value')
    if 'multiply_value' in flat_state:
        state_kwargs['multiply_value'] = flat_state['multiply_value']
    else:
        state_kwargs['multiply_value'] = getattr(model, 'multiply_value')
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

# 1. CUDA source: Fused Transposed Conv + Element-wise Ops
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void deconv_fused_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const float add_value,
    const float multiply_value,
    const int N, const int C_in, const int C_out,
    const int H_in, const int W_in,
    const int H_out, const int W_out,
    const int K, const int stride,
    const int padding)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * C_out * H_out * W_out;
    if (idx >= total) return;

    int tmp = idx;
    const int ow = tmp % W_out; tmp /= W_out;
    const int oh = tmp % H_out; tmp /= H_out;
    const int oc = tmp % C_out; tmp /= C_out;
    const int n  = tmp;

    float sum = 0.0f;
    // Transposed Convolution logic (Stride 2 implementation)
    for (int ic = 0; ic < C_in; ++ic) {
        for (int kh = 0; kh < K; ++kh) {
            int ih_off = oh + padding - kh;
            if (ih_off % stride != 0) continue;
            int ih = ih_off / stride;
            if (ih < 0 || ih >= H_in) continue;

            for (int kw = 0; kw < K; ++kw) {
                int iw_off = ow + padding - kw;
                if (iw_off % stride != 0) continue;
                int iw = iw_off / stride;
                if (iw < 0 || iw >= W_in) continue;

                int in_idx = ((n * C_in + ic) * H_in + ih) * W_in + iw;
                int w_idx = (((oc * C_in + ic) * K + kh) * K + kw);
                sum += input[in_idx] * weight[w_idx];
            }
        }
    }

    if (bias) sum += bias[oc];

    // Fused element-wise: + add_value, min(x,0), gelu, * multiply_value
    float x = fminf(sum + add_value, 0.0f);
    // Approximation for GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    float gelu = 0.5f * x * (1.0f + tanhf(0.79788456f * (x + 0.044715f * x * x * x)));
    output[idx] = gelu * multiply_value;
}

void deconv_fused(
    const torch::Tensor& input, const torch::Tensor& weight, const torch::Tensor& bias,
    torch::Tensor& output, float add_val, float mul_val,
    int N, int C_in, int C_out, int H_in, int W_in, int H_out, int W_out,
    int K, int stride, int padding)
{
    int total = N * C_out * H_out * W_out;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    deconv_fused_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), 
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(), add_val, mul_val,
        N, C_in, C_out, H_in, W_in, H_out, W_out, K, stride, padding);
}
"""

cpp_source = r"""
void deconv_fused(const torch::Tensor& input, const torch::Tensor& weight, const torch::Tensor& bias,
                  torch::Tensor& output, float add_val, float mul_val,
                  int N, int C_in, int C_out, int H_in, int W_in, int H_out, int W_out,
                  int K, int stride, int padding);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("deconv_fused", &deconv_fused);
}
"""

module = load_inline(name='fused_deconv', cpp_sources=cpp_source, cuda_sources=cuda_source, 
                     extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True)

def functional_model(x, *, conv_transpose_weight, conv_transpose_bias, conv_transpose_stride, 
                     conv_transpose_padding, conv_transpose_output_padding, conv_transpose_groups, 
                     conv_transpose_dilation, add_value, multiply_value):
    N, C_in, H_in, W_in = x.shape
    C_out, _, K, _ = conv_transpose_weight.shape
    stride, padding = conv_transpose_stride, conv_transpose_padding
    
    H_out = (H_in - 1) * stride - 2 * padding + K + conv_transpose_output_padding
    W_out = (W_in - 1) * stride - 2 * padding + K + conv_transpose_output_padding
    
    output = torch.empty((N, C_out, H_out, W_out), device=x.device, dtype=torch.float32)
    
    module.deconv_fused(x.contiguous(), conv_transpose_weight.contiguous(), 
                        conv_transpose_bias if conv_transpose_bias is not None else torch.tensor([], device=x.device),
                        output, add_value, multiply_value, N, C_in, C_out, H_in, W_in, H_out, W_out, K, stride, padding)
    return output
