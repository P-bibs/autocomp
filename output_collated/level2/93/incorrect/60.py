# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_160018/code_14.py
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

# ----------------------------------------------------------------------
# CUDA kernel: Manually fused Transposed Convolution + Activation
# Fuses the weight accumulation with the add-min-gelu-mul activation chain
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__device__ __forceinline__ float fast_gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

__global__ void conv_transpose_fused_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const float add_val,
    const float mul_val,
    const int N, const int C_in, const int H_in, const int W_in,
    const int C_out, const int H_out, const int W_out,
    const int kH, const int kW,
    const int stride, const int pad,
    const int groups)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * C_out * H_out * W_out;
    if (idx >= total) return;

    // Decode flat index to (n, oc, oh, ow)
    int tmp = idx;
    const int ow = tmp % W_out; tmp /= W_out;
    const int oh = tmp % H_out; tmp /= H_out;
    const int oc = tmp % C_out; tmp /= C_out;
    const int n  = tmp;

    const int C_in_per_group = C_in / groups;
    const int C_out_per_group = C_out / groups;
    const int group = oc / C_out_per_group;
    const int in_ch_start = group * C_in_per_group;

    float sum = 0.0f;

    // Convolution logic for transposed convolution (kH=4, kW=4)
    // Formula: input_pixel = (output_pixel + padding - k_idx) / stride
    for (int ic = 0; ic < C_in_per_group; ++ic) {
        const int in_c = in_ch_start + ic;
        const int w_offset = (ic * C_out_per_group + (oc % C_out_per_group)) * (kH * kW);
        
        for (int ky = 0; ky < kH; ++ky) {
            int ih_full = oh + pad - ky;
            if (ih_full % stride != 0) continue;
            int ih = ih_full / stride;
            if (ih < 0 || ih >= H_in) continue;

            for (int kx = 0; kx < kW; ++kx) {
                int iw_full = ow + pad - kx;
                if (iw_full % stride != 0) continue;
                int iw = iw_full / stride;
                if (iw < 0 || iw >= W_in) continue;

                sum += input[((n * C_in + in_c) * H_in + ih) * W_in + iw] * 
                       weight[w_offset + (ky * kW + kx)];
            }
        }
    }

    if (bias != nullptr) sum += bias[oc];

    // Activation chain: add -> min(x, 0) -> fast_gelu -> mul
    sum = fast_gelu(fminf(sum + add_val, 0.0f)) * mul_val;

    output[idx] = sum;
}

void conv_transpose_fused(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor output, float add_val, float mul_val,
    int stride, int pad, int groups)
{
    int N = input.size(0), C_in = input.size(1), H_in = input.size(2), W_in = input.size(3);
    int C_out = weight.size(1) * groups, H_out = output.size(2), W_out = output.size(3);
    int kH = weight.size(2), kW = weight.size(3);
    
    int total = N * C_out * H_out * W_out;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    
    float* bias_ptr = (bias.numel() > 0) ? bias.data_ptr<float>() : nullptr;

    conv_transpose_fused_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias_ptr,
        output.data_ptr<float>(), add_val, mul_val,
        N, C_in, H_in, W_in, C_out, H_out, W_out,
        kH, kW, stride, pad, groups);
}
"""

cpp_source = r"""
void conv_transpose_fused(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                          torch::Tensor output, float add_val, float mul_val,
                          int stride, int pad, int groups);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_transpose_fused", &conv_transpose_fused, "Fused Transposed Conv + Activation");
}
"""

fused_ext = load_inline(name='fused_ext', cpp_sources=cpp_source, cuda_sources=cuda_source, 
                        extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True)

def functional_model(x, *, conv_transpose_weight, conv_transpose_bias, 
                     conv_transpose_stride, conv_transpose_padding, 
                     conv_transpose_output_padding, conv_transpose_groups, 
                     conv_transpose_dilation, add_value, multiply_value):
    s = conv_transpose_stride
    p = conv_transpose_padding
    op = conv_transpose_output_padding
    H_out = (x.size(2) - 1) * s - 2 * p + conv_transpose_dilation * (conv_transpose_weight.size(2) - 1) + op + 1
    W_out = (x.size(3) - 1) * s - 2 * p + conv_transpose_dilation * (conv_transpose_weight.size(3) - 1) + op + 1
    out = torch.empty((x.size(0), conv_transpose_weight.size(1) * conv_transpose_groups, H_out, W_out), device='cuda')
    
    fused_ext.conv_transpose_fused(
        x, conv_transpose_weight, 
        conv_transpose_bias if conv_transpose_bias is not None else torch.tensor([], device='cuda'), 
        out, float(add_value), float(multiply_value), s, p, conv_transpose_groups)
    return out
