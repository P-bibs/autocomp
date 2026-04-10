# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_140117/code_31.py
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
import math

# ----------------------------------------------------------------------
# CUDA kernel: fused transposed convolution + element-wise polynomial
# ----------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_deconv_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ elem_bias,
    float* __restrict__ output,
    const int N, const int C_in, const int C_out,
    const int D_in, const int H_in, const int W_in,
    const int D_out, const int H_out, const int W_out,
    const int k_d, const int k_h, const int k_w,
    const int stride, const int padding)
{
    // Total output elements over space and output channels
    int total_elements = N * C_out * D_out * H_out * W_out;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= total_elements) return;

    // Decode linear index to (n, co, d, h, w)
    int rem = idx;
    int w = rem % W_out; rem /= W_out;
    int h = rem % H_out; rem /= H_out;
    int d = rem % D_out; rem /= D_out;
    int co = rem % C_out; rem /= C_out;
    int n = rem;

    float acc = conv_bias[co];

    // Transposed Conv Logic: 
    // For each output pixel, sum contributions from kernel weights and input pixels
    for (int ci = 0; ci < C_in; ++ci) {
        for (int kd = 0; kd < k_d; ++kd) {
            int di = d + padding - kd;
            if (di % stride != 0) continue;
            di /= stride;
            if (di < 0 || di >= D_in) continue;

            for (int kh = 0; kh < k_h; ++kh) {
                int hi = h + padding - kh;
                if (hi % stride != 0) continue;
                hi /= stride;
                if (hi < 0 || hi >= H_in) continue;

                for (int kw = 0; kw < k_w; ++kw) {
                    int wi = w + padding - kw;
                    if (wi % stride != 0) continue;
                    wi /= stride;
                    if (wi < 0 || wi >= W_in) continue;

                    // weight layout: (C_in, C_out, k_d, k_h, k_w)
                    int w_idx = (((ci * C_out + co) * k_d + kd) * k_h + kh) * k_w + kw;
                    int in_idx = (((n * C_in + ci) * D_in + di) * H_in + hi) * W_in + wi;
                    acc += weight[w_idx] * input[in_idx];
                }
            }
        }
    }

    // Apply polynomial: ((x + bias) + x) * x + x = x*(2*x + bias + 1)
    float b = elem_bias[co];
    output[idx] = acc * (2.0f * acc + b + 1.0f);
}

void fused_deconv_forward(
    const torch::Tensor& input, const torch::Tensor& weight, 
    const torch::Tensor& conv_bias, const torch::Tensor& elem_bias, 
    torch::Tensor& output,
    int N, int C_in, int C_out, 
    int D_in, int H_in, int W_in, 
    int D_out, int H_out, int W_out,
    int k_d, int k_h, int k_w, int stride, int padding)
{
    int total_elements = N * C_out * D_out * H_out * W_out;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    
    fused_deconv_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), 
        conv_bias.data_ptr<float>(), elem_bias.data_ptr<float>(), 
        output.data_ptr<float>(),
        N, C_in, C_out, D_in, H_in, W_in, D_out, H_out, W_out,
        k_d, k_h, k_w, stride, padding
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_deconv_forward(const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, torch::Tensor&, int, int, int, int, int, int, int, int, int, int, int, int, int, int);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_deconv", &fused_deconv_forward, "Fused Transposed Conv + Poly");
}
"""

fused_ext = load_inline(
    name='fused_deconv_ext', cpp_sources=cpp_source, cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True
)

def functional_model(
    x, *,
    conv_transpose_weight, conv_transpose_bias,
    conv_transpose_stride, conv_transpose_padding,
    conv_transpose_output_padding, conv_transpose_groups,
    conv_transpose_dilation, bias
):
    N, C_in, D, H, W = x.shape
    C_out, _, kd, kh, kw = conv_transpose_weight.shape
    
    D_out = (D - 1) * conv_transpose_stride - 2 * conv_transpose_padding + (kd - 1) + conv_transpose_output_padding + 1
    H_out = (H - 1) * conv_transpose_stride - 2 * conv_transpose_padding + (kh - 1) + conv_transpose_output_padding + 1
    W_out = (W - 1) * conv_transpose_stride - 2 * conv_transpose_padding + (kw - 1) + conv_transpose_output_padding + 1
    
    output = torch.empty((N, C_out, D_out, H_out, W_out), device='cuda', dtype=torch.float32)
    
    fused_ext.fused_deconv(
        x.contiguous(), conv_transpose_weight.contiguous(), 
        conv_transpose_bias.view(-1), bias.view(-1), output,
        N, C_in, C_out, D, H, W, D_out, H_out, W_out,
        kd, kh, kw, conv_transpose_stride, conv_transpose_padding
    )
    return output
