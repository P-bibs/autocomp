# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_144547/code_3.py
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
# 1.  CUDA source for the convolution kernel
# ----------------------------------------------------------------------
conv_cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_forward_kernel(
    const float* __restrict__ inp,
    const float* __restrict__ w,
    const float* __restrict__ bias,
    float* out,
    const int N, const int C_in, const int H, const int W,
    const int C_out, const int groups,
    const int kH, const int kW,
    const int stride, const int pad, const int dil,
    const int out_h, const int out_w)
{
    const int n = blockIdx.x;                         // batch index
    const int co = blockIdx.y * blockDim.x + threadIdx.x; // output channel
    const int spatial = blockIdx.z;                    // flattened (y,x)
    const int y = spatial / out_w;
    const int x = spatial % out_w;

    if (co >= C_out || y >= out_h || x >= out_w) return;

    const int g_out_ch = C_out / groups;
    const int g_in_ch  = C_in  / groups;
    const int g_id = co / g_out_ch;               // which group this thread belongs to
    const int co_loc = co % g_out_ch;             // channel inside the group

    float sum = 0.0f;

    // loops over kernel spatial positions
    for (int ky = 0; ky < kH; ++ky) {
        int iy = y * stride - pad + ky * dil;
        if (iy < 0 || iy >= H) continue;
        for (int kx = 0; kx < kW; ++kx) {
            int ix = x * stride - pad + kx * dil;
            if (ix < 0 || ix >= W) continue;
            // loop over input channels inside the group
            for (int ci = 0; ci < g_in_ch; ++ci) {
                int in_ch = g_id * g_in_ch + ci;
                int inp_idx = ((n * C_in + in_ch) * H + iy) * W + ix;
                int w_idx   = ((co_loc * g_in_ch + ci) * kH + ky) * kW + kx;
                sum += inp[inp_idx] * w[w_idx];
            }
        }
    }

    if (bias) sum += bias[co];

    int out_idx = ((n * C_out + co) * out_h + y) * out_w + x;
    out[out_idx] = sum;
}

torch::Tensor conv_forward_cuda(
    torch::Tensor inp, torch::Tensor weight, torch::Tensor bias,
    int stride, int pad, int dil, int groups,
    int out_h, int out_w) {
    
    const int N = inp.size(0);
    const int C_in = inp.size(1);
    const int H = inp.size(2);
    const int W = inp.size(3);
    const int C_out = weight.size(0);
    const int kH = weight.size(2);
    const int kW = weight.size(3);

    auto out = torch::empty({N, C_out, out_h, out_w}, 
                            torch::TensorOptions().dtype(inp.dtype()).device(inp.device()));

    dim3 blocks(N, (C_out + 31) / 32, out_h * out_w);
    dim3 threads(32);

    conv_forward_kernel<<<blocks, threads>>>(
        inp.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        out.data_ptr<float>(),
        N, C_in, H, W,
        C_out, groups,
        kH, kW,
        stride, pad, dil,
        out_h, out_w
    );

    return out;
}
"""

# ----------------------------------------------------------------------
# 2.  CUDA source for the fused scale + max‑pool + clamp kernel
# ----------------------------------------------------------------------
fused_cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_scale_maxpool_clamp_kernel(
    const float* __restrict__ inp,
    const float* __restrict__ scale,
    float* __restrict__ out,
    const int N, const int C, const int H, const int W,
    const int kH, const int kW,
    const int stride, const int pad, const int dil,
    const int out_h, const int out_w,
    const float clamp_min, const float clamp_max)
{
    const int n   = blockIdx.x;
    const int c   = blockIdx.y * blockDim.x + threadIdx.x;
    const int y   = blockIdx.z / out_w;
    const int x   = blockIdx.z % out_w;

    if (c >= C || y >= out_h || x >= out_w) return;

    float maxv = -1e38f;   // -infinity

    for (int ky = 0; ky < kH; ++ky) {
        int iy = y * stride - pad + ky * dil;
        if (iy < 0 || iy >= H) continue;
        for (int kx = 0; kx < kW; ++kx) {
            int ix = x * stride - pad + kx * dil;
            if (ix < 0 || ix >= W) continue;
            int idx = ((n * C + c) * H + iy) * W + ix;
            float v = inp[idx];
            if (v > maxv) maxv = v;
        }
    }

    // per‑channel scale
    float scaled = maxv * scale[c];

    // clamp
    if (scaled < clamp_min) scaled = clamp_min;
    else if (scaled > clamp_max) scaled = clamp_max;

    int out_idx = ((n * C + c) * out_h + y) * out_w + x;
    out[out_idx] = scaled;
}

torch::Tensor fused_scale_maxpool_clamp_cuda(
    torch::Tensor inp, torch::Tensor scale,
    int kH, int kW, int stride, int pad, int dil,
    int out_h, int out_w,
    float clamp_min, float clamp_max) {
    
    const int N = inp.size(0);
    const int C = inp.size(1);
    const int H = inp.size(2);
    const int W = inp.size(3);

    auto out = torch::empty({N, C, out_h, out_w}, 
                            torch::TensorOptions().dtype(inp.dtype()).device(inp.device()));

    dim3 blocks(N, (C + 31) / 32, out_h * out_w);
    dim3 threads(32);

    fused_scale_maxpool_clamp_kernel<<<blocks, threads>>>(
        inp.data_ptr<float>(),
        scale.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C, H, W,
        kH, kW,
        stride, pad, dil,
        out_h, out_w,
        clamp_min, clamp_max
    );

    return out;
}
"""

# ----------------------------------------------------------------------
# 3.  C++ wrapper that launches the kernels (PYBIND11 binding)
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

torch::Tensor conv_forward_cuda(
    torch::Tensor inp, torch::Tensor weight, torch::Tensor bias,
    int stride, int pad, int dil, int groups,
    int out_h, int out_w);

torch::Tensor fused_scale_maxpool_clamp_cuda(
    torch::Tensor inp, torch::Tensor scale,
    int kH, int kW, int stride, int pad, int dil,
    int out_h, int out_w,
    float clamp_min, float clamp_max);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_forward", &conv_forward_cuda, "conv forward");
    m.def("fused_scale_maxpool_clamp", &fused_scale_maxpool_clamp_cuda, "fused scale+maxpool+clamp");
}
"""

# ----------------------------------------------------------------------
# 4.  Build the inline extension
# ----------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_ops',
    cpp_sources=cpp_source,
    cuda_sources=conv_cuda_source + fused_cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True,
)

# ----------------------------------------------------------------------
# 5.  Helper to compute output spatial size of convolution and max‑pool
# ----------------------------------------------------------------------
def conv_out_size(H, W, kernel, stride, pad, dil):
    return (H + 2 * pad - dil * (kernel - 1) - 1) // stride + 1, \
           (W + 2 * pad - dil * (kernel - 1) - 1) // stride + 1

def maxpool_out_size(H, W, kernel, stride, pad, dil, ceil_mode):
    if ceil_mode:
        return (H + 2 * pad - dil * (kernel - 1) + stride - 1) // stride, \
               (W + 2 * pad - dil * (kernel - 1) + stride - 1) // stride
    else:
        return (H + 2 * pad - dil * (kernel - 1) - 1) // stride + 1, \
               (W + 2 * pad - dil * (kernel - 1) - 1) // stride + 1

# ----------------------------------------------------------------------
# 6.  The functional model used by the evaluator
# ----------------------------------------------------------------------
def functional_model(
    x,
    *,
    conv_weight,
    conv_bias,
    conv_stride,
    conv_padding,
    conv_dilation,
    conv_groups,
    group_norm_weight,
    group_norm_bias,
    group_norm_num_groups,
    group_norm_eps,
    maxpool_kernel_size,
    maxpool_stride,
    maxpool_padding,
    maxpool_dilation,
    maxpool_ceil_mode,
    maxpool_return_indices,
    scale,
    clamp_min,
    clamp_max,
):
    # ------------------------------------------------------------------
    # 1) custom convolution
    # ------------------------------------------------------------------
    N, C_in, H, W = x.shape
    C_out = conv_weight.shape[0]
    kernel_size = conv_weight.shape[2]          # square kernel assumed

    out_h, out_w = conv_out_size(H, W, kernel_size,
                                 conv_stride, conv_padding, conv_dilation)

    # launch conv kernel
    conv_out = fused_ext.conv_forward(
        x, conv_weight, conv_bias,
        conv_stride, conv_padding, conv_dilation, conv_groups,
        out_h, out_w
    )

    # ------------------------------------------------------------------
    # 2) group norm (keep PyTorch's optimized implementation)
    # ------------------------------------------------------------------
    x = F.group_norm(conv_out, group_norm_num_groups,
                     weight=group_norm_weight, bias=group_norm_bias,
                     eps=group_norm_eps)

    # ------------------------------------------------------------------
    # 3) fused scale + max‑pool + clamp
    # ------------------------------------------------------------------
    # shape after group norm is unchanged: (N, C_out, out_h, out_w)
    N2, C2, H2, W2 = x.shape
    pool_out_h, pool_out_w = maxpool_out_size(
        H2, W2, maxpool_kernel_size,
        maxpool_stride, maxpool_padding,
        maxpool_dilation, maxpool_ceil_mode)

    # scale must be a 1‑D tensor of length C2
    scale_1d = scale.squeeze().contiguous()

    result = fused_ext.fused_scale_maxpool_clamp(
        x, scale_1d,
        maxpool_kernel_size, maxpool_kernel_size,
        maxpool_stride, maxpool_padding, maxpool_dilation,
        pool_out_h, pool_out_w,
        clamp_min, clamp_max)

    return result

# ----------------------------------------------------------------------
# 7.  Dummy parameters used by the test harness (not part of the model)
# ----------------------------------------------------------------------
batch_size = 128
in_channels = 8
out_channels = 64
height, width = 128, 128
kernel_size = 3
num_groups = 16
scale_shape = (out_channels, 1, 1)
maxpool_kernel_size = 4
clamp_min = 0.0
clamp_max = 1.0

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, num_groups,
            scale_shape, maxpool_kernel_size, clamp_min, clamp_max]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]
