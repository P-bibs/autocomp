# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_115905/code_22.py
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

# -------------------------------------------------------------------------
# CUDA source – Fused Transposed Convolution + Bias Subtraction + Tanh
# -------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_transpose_conv_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ act_bias,
    float* __restrict__ output,
    const int N, const int C_in, const int C_out,
    const int H_in, const int W_in,
    const int H_out, const int W_out,
    const int kH, const int kW,
    const int stride, const int padding)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_pixels = N * C_out * H_out * W_out;
    if (idx >= total_pixels) return;

    // Decode flattened index
    int tmp = idx;
    const int w_out = tmp % W_out; tmp /= W_out;
    const int h_out = tmp % H_out; tmp /= H_out;
    const int co = tmp % C_out; tmp /= C_out;
    const int n = tmp;

    float sum = 0.0f;

    // Transposed convolution: iterate over kernel and input
    for (int ci = 0; ci < C_in; ++ci) {
        for (int kh = 0; kh < kH; ++kh) {
            int h_in = h_out + padding - kh;
            if (h_in % stride != 0) continue;
            h_in /= stride;
            if (h_in < 0 || h_in >= H_in) continue;

            for (int kw = 0; kw < kW; ++kw) {
                int w_in = w_out + padding - kw;
                if (w_in % stride != 0) continue;
                w_in /= stride;
                if (w_in < 0 || w_in >= W_in) continue;

                // Weight layout: (C_out, C_in, kH, kW)
                float wVal = weight[((co * C_in + ci) * kH + kh) * kW + kw];
                // Input layout: (N, C_in, H_in, W_in)
                float iVal = input[((n * C_in + ci) * H_in + h_in) * W_in + w_in];
                sum += wVal * iVal;
            }
        }
    }

    sum += conv_bias[co];
    output[idx] = tanhf(sum - act_bias[co]);
}

void fused_transpose_conv(
    torch::Tensor input, torch::Tensor weight, torch::Tensor conv_bias,
    torch::Tensor act_bias, torch::Tensor output,
    int stride, int padding)
{
    const int N = input.size(0);
    const int C_in = input.size(1);
    const int H_in = input.size(2);
    const int W_in = input.size(3);
    const int C_out = weight.size(0);
    const int kH = weight.size(2);
    const int kW = weight.size(3);
    const int H_out = output.size(2);
    const int W_out = output.size(3);

    const int total = N * C_out * H_out * W_out;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    fused_transpose_conv_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(),
        conv_bias.data_ptr<float>(), act_bias.data_ptr<float>(),
        output.data_ptr<float>(), N, C_in, C_out, H_in, W_in,
        H_out, W_out, kH, kW, stride, padding
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_transpose_conv(torch::Tensor input, torch::Tensor weight, torch::Tensor conv_bias, torch::Tensor act_bias, torch::Tensor output, int stride, int padding);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_transpose_conv", &fused_transpose_conv, "Fused Transposed Conv");
}
"""

fused_ext = load_inline(
    name='fused_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(
    x, *, conv_transpose_weight, conv_transpose_bias, conv_transpose_stride,
    conv_transpose_padding, conv_transpose_output_padding, conv_transpose_groups,
    conv_transpose_dilation, bias,
):
    # Calculate output dimensions
    N, C_in, H_in, W_in = x.shape
    C_out, _, kH, kW = conv_transpose_weight.shape
    H_out = (H_in - 1) * conv_transpose_stride - 2 * conv_transpose_padding + (kH - 1) + conv_transpose_output_padding + 1
    W_out = (W_in - 1) * conv_transpose_stride - 2 * conv_transpose_padding + (kW - 1) + conv_transpose_output_padding + 1
    
    output = torch.empty((N, C_out, H_out, W_out), device='cuda', dtype=torch.float32)
    
    fused_ext.fused_transpose_conv(
        x.contiguous(), 
        conv_transpose_weight.contiguous(), 
        conv_transpose_bias.contiguous(), 
        bias.view(-1).contiguous(), 
        output, 
        conv_transpose_stride, 
        conv_transpose_padding
    )
    return output
