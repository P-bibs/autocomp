# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_144547/code_7.py
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
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# ----------------------------------------------------------------------
# CUDA Kernels
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_forward_kernel(
    const float* __restrict__ inp, const float* __restrict__ w, const float* __restrict__ bias,
    float* __restrict__ out, const int N, const int C_in, const int H, const int W,
    const int C_out, const int groups, const int kH, const int kW,
    const int stride, const int pad, const int dil, const int out_h, const int out_w)
{
    const int n = blockIdx.x;
    const int co = blockIdx.y * blockDim.x + threadIdx.x;
    const int y = blockIdx.z / out_w;
    const int x = blockIdx.z % out_w;

    if (co >= C_out || y >= out_h || x >= out_w) return;

    const int g_out_ch = C_out / groups;
    const int g_in_ch = C_in / groups;
    const int g_id = co / g_out_ch;
    const int co_loc = co % g_out_ch;

    float sum = 0.0f;
    for (int ky = 0; ky < kH; ++ky) {
        int iy = y * stride - pad + ky * dil;
        if (iy < 0 || iy >= H) continue;
        for (int kx = 0; kx < kW; ++kx) {
            int ix = x * stride - pad + kx * dil;
            if (ix < 0 || ix >= W) continue;
            for (int ci = 0; ci < g_in_ch; ++ci) {
                int in_ch = g_id * g_in_ch + ci;
                sum += inp[((n * C_in + in_ch) * H + iy) * W + ix] * 
                       w[((co_loc * g_in_ch + ci) * kH + ky) * kW + kx];
            }
        }
    }
    if (bias) sum += bias[co];
    out[((n * C_out + co) * out_h + y) * out_w + x] = sum;
}

__global__ void fused_pool_scale_clamp_kernel(
    const float* __restrict__ inp, const float* __restrict__ scale,
    float* __restrict__ out, const int N, const int C, const int H, const int W,
    const int kH, const int kW, const int stride, const int pad, const int dil,
    const int out_h, const int out_w, const float c_min, const float c_max)
{
    const int n = blockIdx.x;
    const int c = blockIdx.y * blockDim.x + threadIdx.x;
    const int y = blockIdx.z / out_w;
    const int x = blockIdx.z % out_w;

    if (c >= C || y >= out_h || x >= out_w) return;

    float maxv = -1e38f;
    for (int ky = 0; ky < kH; ++ky) {
        int iy = y * stride - pad + ky * dil;
        if (iy < 0 || iy >= H) continue;
        for (int kx = 0; kx < kW; ++kx) {
            int ix = x * stride - pad + kx * dil;
            if (ix < 0 || ix >= W) continue;
            float v = inp[((n * C + c) * H + iy) * W + ix];
            if (v > maxv) maxv = v;
        }
    }
    float val = maxv * scale[c];
    out[((n * C + c) * out_h + y) * out_w + x] = (val < c_min) ? c_min : ((val > c_max) ? c_max : val);
}

void conv_forward(at::Tensor inp, at::Tensor w, at::Tensor b, int s, int p, int d, int g, at::Tensor out) {
    int N = inp.size(0), C_in = inp.size(1), H = inp.size(2), W = inp.size(3);
    int C_out = w.size(0), out_h = out.size(2), out_w = out.size(3), kH = w.size(2);
    dim3 threads(32);
    dim3 blocks(N, (C_out + 31) / 32, out_h * out_w);
    conv_forward_kernel<<<blocks, threads>>>(inp.data_ptr<float>(), w.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), N, C_in, H, W, C_out, g, kH, kH, s, p, d, out_h, out_w);
}

void fused_pool(at::Tensor inp, at::Tensor scale, at::Tensor out, int kH, int s, int p, int d, float c_min, float c_max) {
    int N = inp.size(0), C = inp.size(1), H = inp.size(2), W = inp.size(3);
    int out_h = out.size(2), out_w = out.size(3);
    dim3 threads(32);
    dim3 blocks(N, (C + 31) / 32, out_h * out_w);
    fused_pool_scale_clamp_kernel<<<blocks, threads>>>(inp.data_ptr<float>(), scale.data_ptr<float>(), out.data_ptr<float>(), N, C, H, W, kH, kH, s, p, d, out_h, out_w, c_min, c_max);
}
"""

cpp_source = r"""
void conv_forward(at::Tensor i, at::Tensor w, at::Tensor b, int s, int p, int d, int g, at::Tensor o);
void fused_pool(at::Tensor i, at::Tensor s, at::Tensor o, int kH, int st, int p, int d, float cmin, float cmax);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_forward", &conv_forward);
    m.def("fused_pool", &fused_pool);
}
"""

fused_ext = load_inline(name='fused_ops', cpp_sources=cpp_source, cuda_sources=cuda_source, extra_cuda_cflags=['-O3'], with_cuda=True)

def functional_model(x, *, conv_weight, conv_bias, conv_stride, conv_padding, conv_dilation, conv_groups, group_norm_weight, group_norm_bias, group_norm_num_groups, group_norm_eps, maxpool_kernel_size, maxpool_stride, maxpool_padding, maxpool_dilation, maxpool_ceil_mode, maxpool_return_indices, scale, clamp_min, clamp_max):
    N, C_in, H, W = x.shape
    C_out = conv_weight.size(0)
    out_h = (H + 2 * conv_padding - conv_dilation * (conv_weight.size(2) - 1) - 1) // conv_stride + 1
    out_w = (W + 2 * conv_padding - conv_dilation * (conv_weight.size(3) - 1) - 1) // conv_stride + 1
    out_conv = torch.empty((N, C_out, out_h, out_w), device=x.device, dtype=x.dtype)
    fused_ext.conv_forward(x, conv_weight, conv_bias, conv_stride, conv_padding, conv_dilation, conv_groups, out_conv)
    norm = F.group_norm(out_conv, group_norm_num_groups, group_norm_weight, group_norm_bias, group_norm_eps)
    p_h = (out_h + 2 * maxpool_padding - maxpool_dilation * (maxpool_kernel_size - 1) - 1) // maxpool_stride + 1
    p_w = (out_w + 2 * maxpool_padding - maxpool_dilation * (maxpool_kernel_size - 1) - 1) // maxpool_stride + 1
    out_final = torch.empty((N, C_out, p_h, p_w), device=x.device, dtype=x.dtype)
    fused_ext.fused_pool(norm, scale.flatten(), out_final, maxpool_kernel_size, maxpool_stride, maxpool_padding, maxpool_dilation, clamp_min, clamp_max)
    return out_final
