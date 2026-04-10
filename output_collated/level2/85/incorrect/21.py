# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_143758/code_7.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'num_groups', 'scale_shape', 'maxpool_kernel_size', 'clamp_min', 'clamp_max']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_weight', 'conv_bias', 'conv_stride', 'conv_padding', 'conv_dilation', 'conv_groups', 'group_norm_weight', 'group_norm_bias', 'group_norm_num_groups', 'group_norm_eps', 'maxpool_kernel_size', 'maxpool_stride', 'maxpool_padding', 'maxpool_dilation', 'maxpool_ceil_mode', 'maxpool_return_indices', 'scale', 'clamp_min', 'clamp_max']
REQUIRED_FLAT_STATE_NAMES = ['conv_weight', 'conv_bias', 'group_norm_weight', 'group_norm_bias', 'scale']


class ModelNew(nn.Module):
    """
    ModelNew that performs convolution, group normalization, scaling, max pooling, and clamping.
    """

    def __init__(self, in_channels, out_channels, kernel_size, num_groups, scale_shape, maxpool_kernel_size, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.scale = nn.Parameter(torch.ones(scale_shape))
        self.maxpool = nn.MaxPool2d(kernel_size=maxpool_kernel_size)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

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
    # State for conv (nn.Conv2d)
    if 'conv_weight' in flat_state:
        state_kwargs['conv_weight'] = flat_state['conv_weight']
    else:
        state_kwargs['conv_weight'] = getattr(model.conv, 'weight', None)
    if 'conv_bias' in flat_state:
        state_kwargs['conv_bias'] = flat_state['conv_bias']
    else:
        state_kwargs['conv_bias'] = getattr(model.conv, 'bias', None)
    state_kwargs['conv_stride'] = model.conv.stride
    state_kwargs['conv_padding'] = model.conv.padding
    state_kwargs['conv_dilation'] = model.conv.dilation
    state_kwargs['conv_groups'] = model.conv.groups
    # State for group_norm (nn.GroupNorm)
    if 'group_norm_weight' in flat_state:
        state_kwargs['group_norm_weight'] = flat_state['group_norm_weight']
    else:
        state_kwargs['group_norm_weight'] = getattr(model.group_norm, 'weight', None)
    if 'group_norm_bias' in flat_state:
        state_kwargs['group_norm_bias'] = flat_state['group_norm_bias']
    else:
        state_kwargs['group_norm_bias'] = getattr(model.group_norm, 'bias', None)
    state_kwargs['group_norm_num_groups'] = model.group_norm.num_groups
    state_kwargs['group_norm_eps'] = model.group_norm.eps
    # State for maxpool (nn.MaxPool2d)
    state_kwargs['maxpool_kernel_size'] = model.maxpool.kernel_size
    state_kwargs['maxpool_stride'] = model.maxpool.stride
    state_kwargs['maxpool_padding'] = model.maxpool.padding
    state_kwargs['maxpool_dilation'] = model.maxpool.dilation
    state_kwargs['maxpool_ceil_mode'] = model.maxpool.ceil_mode
    state_kwargs['maxpool_return_indices'] = model.maxpool.return_indices
    if 'scale' in flat_state:
        state_kwargs['scale'] = flat_state['scale']
    else:
        state_kwargs['scale'] = getattr(model, 'scale')
    if 'clamp_min' in flat_state:
        state_kwargs['clamp_min'] = flat_state['clamp_min']
    else:
        state_kwargs['clamp_min'] = getattr(model, 'clamp_min')
    if 'clamp_max' in flat_state:
        state_kwargs['clamp_max'] = flat_state['clamp_max']
    else:
        state_kwargs['clamp_max'] = getattr(model, 'clamp_max')
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
# CUDA Kernel implementation
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <algorithm>

// 1) Direct convolution kernel (replaces torch.conv2d)
__global__ void conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C_in, int H, int W,
    int C_out, int H_out, int W_out,
    int k, int stride, int padding, int dilation)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_out * H_out * W_out;
    if (idx >= total) return;

    int n = idx / (C_out * H_out * W_out);
    int rem = idx % (C_out * H_out * W_out);
    int oc = rem / (H_out * W_out);
    int rem2 = rem % (H_out * W_out);
    int oh = rem2 / W_out;
    int ow = rem2 % W_out;

    float val = bias[oc];
    int h_base = oh * stride - padding;
    int w_base = ow * stride - padding;

    for (int ic = 0; ic < C_in; ++ic) {
        for (int kh = 0; kh < k; ++kh) {
            int ih = h_base + kh * dilation;
            if (ih < 0 || ih >= H) continue;
            for (int kw = 0; kw < k; ++kw) {
                int iw = w_base + kw * dilation;
                if (iw < 0 || iw >= W) continue;
                
                float in_val = input[((n * C_in + ic) * H + ih) * W + iw];
                float w_val = weight[(((oc * C_in + ic) * k) + kh) * k + kw];
                val += in_val * w_val;
            }
        }
    }
    output[idx] = val;
}

// 2) Fused Max-Pool + Scale + Clamp kernel
__global__ void fused_pool_scale_clamp_kernel(
    const float* __restrict__ input,
    const float* __restrict__ scale,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int H_out, int W_out,
    int k, int stride, int padding, int dilation,
    float c_min, float c_max)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H_out * W_out;
    if (idx >= total) return;

    int n = idx / (C * H_out * W_out);
    int rem = idx % (C * H_out * W_out);
    int c = rem / (H_out * W_out);
    int rem2 = rem % (H_out * W_out);
    int oh = rem2 / W_out;
    int ow = rem2 % W_out;

    int h_start = oh * stride - padding;
    int w_start = ow * stride - padding;

    float mx = -1e38f;
    for (int kh = 0; kh < k; ++kh) {
        int ih = h_start + kh * dilation;
        if (ih < 0 || ih >= H) continue;
        for (int kw = 0; kw < k; ++kw) {
            int iw = w_start + kw * dilation;
            if (iw < 0 || iw >= W) continue;
            float val = input[((n * C + c) * H + ih) * W + iw];
            if (val > mx) mx = val;
        }
    }

    mx *= scale[c];
    output[idx] = fminf(fmaxf(mx, c_min), c_max);
}
"""

cpp_source = r"""
#include <torch/extension.h>
void conv_kernel(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output, int stride, int padding, int dilation);
void pool_kernel(torch::Tensor input, torch::Tensor scale, torch::Tensor output, int k, int stride, int padding, int dilation, float c_min, float c_max);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv", &conv_kernel);
    m.def("pool", &pool_kernel);
}
"""

# Wrapper logic to handle kernel launches
cuda_helpers = """
void conv_kernel(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output, int stride, int padding, int dilation) {
    int N = input.size(0); int C_in = input.size(1); int H = input.size(2); int W = input.size(3);
    int C_out = weight.size(0); int H_out = output.size(2); int W_out = output.size(3);
    int k = weight.size(2);
    int total = N * C_out * H_out * W_out;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    conv2d_kernel<<<blocks, threads>>>(input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(), N, C_in, H, W, C_out, H_out, W_out, k, stride, padding, dilation);
}
void pool_kernel(torch::Tensor input, torch::Tensor scale, torch::Tensor output, int k, int stride, int padding, int dilation, float c_min, float c_max) {
    int N = input.size(0); int C = input.size(1); int H = input.size(2); int W = input.size(3);
    int H_out = output.size(2); int W_out = output.size(3);
    int total = N * C * H_out * W_out;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    fused_pool_scale_clamp_kernel<<<blocks, threads>>>(input.data_ptr<float>(), scale.data_ptr<float>(), output.data_ptr<float>(), N, C, H, W, H_out, W_out, k, stride, padding, dilation, c_min, c_max);
}
"""

fused_ext = load_inline(name='fused_op', cpp_sources=cpp_source, cuda_sources=cuda_source + cuda_helpers, extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True)

def functional_model(x, *, conv_weight, conv_bias, conv_stride, conv_padding, conv_dilation, conv_groups, group_norm_weight, group_norm_bias, group_norm_num_groups, group_norm_eps, maxpool_kernel_size, maxpool_stride, maxpool_padding, maxpool_dilation, maxpool_ceil_mode, maxpool_return_indices, scale, clamp_min, clamp_max):
    # Output dims
    k = conv_weight.size(2)
    H_out = (x.size(2) + 2 * conv_padding - conv_dilation * (k - 1) - 1) // conv_stride + 1
    W_out = (x.size(3) + 2 * conv_padding - conv_dilation * (k - 1) - 1) // conv_stride + 1
    conv_out = torch.empty((x.size(0), conv_weight.size(0), H_out, W_out), device=x.device, dtype=x.dtype)
    fused_ext.conv(x.contiguous(), conv_weight.contiguous(), conv_bias.contiguous(), conv_out, conv_stride, conv_padding, conv_dilation)
    
    x = torch.nn.functional.group_norm(conv_out, group_norm_num_groups, group_norm_weight, group_norm_bias, group_norm_eps)
    
    H_pool = (x.size(2) + 2 * maxpool_padding - maxpool_dilation * (maxpool_kernel_size - 1) - 1) // maxpool_stride + 1
    W_pool = (x.size(3) + 2 * maxpool_padding - maxpool_dilation * (maxpool_kernel_size - 1) - 1) // maxpool_stride + 1
    pool_out = torch.empty((x.size(0), x.size(1), H_pool, W_pool), device=x.device, dtype=x.dtype)
    fused_ext.pool(x.contiguous(), scale.view(-1).contiguous(), pool_out, maxpool_kernel_size, maxpool_stride, maxpool_padding, maxpool_dilation, clamp_min, clamp_max)
    return pool_out

def get_init_inputs(): return [8, 64, 3, 16, (64, 1, 1), 4, 0.0, 1.0]
def get_inputs(): return [torch.rand(128, 8, 128, 128).cuda()]
