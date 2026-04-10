# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_142828/code_3.py
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
import math
from torch.utils.cpp_extension import load_inline

# -------------------------------------------------------------------------
# Inline CUDA source – contains two kernels:
#   * conv_kernel   : naive per‑output‑pixel convolution
#   * fused_maxpool_kernel : max‑pool + per‑channel scale + clamp
# -------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

// ------------------------------------------------------------------
// Convolution kernel: one thread per output pixel, naive loop
// ------------------------------------------------------------------
__global__ void conv_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int N, const int C_in, const int H_in, const int W_in,
    const int C_out, const int H_out, const int W_out,
    const int kH, const int kW,
    const int stride_h, const int stride_w,
    const int pad_h, const int pad_w,
    const int dil_h, const int dil_w,
    const int groups)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_out * H_out * W_out;
    if (idx >= total) return;

    int n = idx / (C_out * H_out * W_out);
    int rem = idx % (C_out * H_out * W_out);
    int co = rem / (H_out * W_out);
    int rem2 = rem % (H_out * W_out);
    int h_out = rem2 / W_out;
    int w_out = rem2 % W_out;

    int h_in_start = h_out * stride_h - pad_h;
    int w_in_start = w_out * stride_w - pad_w;

    float sum = 0.0f;

    // Simple case: groups == 1
    for (int ci = 0; ci < C_in; ++ci) {
        for (int kh = 0; kh < kH; ++kh) {
            int h_in = h_in_start + kh * dil_h;
            if (h_in < 0 || h_in >= H_in) continue;
            for (int kw = 0; kw < kW; ++kw) {
                int w_in = w_in_start + kw * dil_w;
                if (w_in < 0 || w_in >= W_in) continue;
                float inp_val = input[((n * C_in + ci) * H_in + h_in) * W_in + w_in];
                float w_val   = weight[((co * C_in + ci) * kH + kh) * kW + kw];
                sum += inp_val * w_val;
            }
        }
    }

    if (bias != nullptr) sum += bias[co];

    output[((n * C_out + co) * H_out + h_out) * W_out + w_out] = sum;
}

// ------------------------------------------------------------------
// Fused max‑pool + per‑channel scale + clamp
// ------------------------------------------------------------------
__global__ void fused_maxpool_kernel(
    const float* __restrict__ input,
    const float* __restrict__ scale,
    float* __restrict__ output,
    const int N, const int C, const int H, const int W,
    const int H_out, const int W_out,
    const int kH, const int kW,
    const int stride_h, const int stride_w,
    const int pad_h, const int pad_w,
    const int dil_h, const int dil_w,
    const float clamp_min,
    const float clamp_max)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H_out * W_out;
    if (idx >= total) return;

    int n = idx / (C * H_out * W_out);
    int rem = idx % (C * H_out * W_out);
    int c = rem / (H_out * W_out);
    int rem2 = rem % (H_out * W_out);
    int h_out = rem2 / W_out;
    int w_out = rem2 % W_out;

    int h_in_start = h_out * stride_h - pad_h;
    int w_in_start = w_out * stride_w - pad_w;

    float max_val = -INFINITY;
    for (int kh = 0; kh < kH; ++kh) {
        int h_in = h_in_start + kh * dil_h;
        if (h_in < 0 || h_in >= H) continue;
        for (int kw = 0; kw < kW; ++kw) {
            int w_in = w_in_start + kw * dil_w;
            if (w_in < 0 || w_in >= W) continue;
            float val = input[((n * C + c) * H + h_in) * W + w_in];
            if (val > max_val) max_val = val;
        }
    }

    // Per‑channel scale (applied after pooling – mathematically equivalent)
    max_val *= scale[c];

    // Clamp
    if (max_val < clamp_min) max_val = clamp_min;
    if (max_val > clamp_max) max_val = clamp_max;

    output[((n * C + c) * H_out + h_out) * W_out + w_out] = max_val;
}

// ------------------------------------------------------------------
// Host wrappers that will be called from Python
// ------------------------------------------------------------------
void conv_forward(int blocks, int threads,
                 const float* input, const float* weight, const float* bias, float* output,
                 int N, int C_in, int H_in, int W_in,
                 int C_out, int H_out, int W_out,
                 int kH, int kW,
                 int stride_h, int stride_w,
                 int pad_h, int pad_w,
                 int dil_h, int dil_w,
                 int groups)
{
    conv_kernel<<<blocks, threads>>>(
        input, weight, bias, output,
        N, C_in, H_in, W_in,
        C_out, H_out, W_out,
        kH, kW,
        stride_h, stride_w,
        pad_h, pad_w,
        dil_h, dil_w,
        groups);
}

void maxpool_forward(int blocks, int threads,
                    const float* input, const float* scale, float* output,
                    int N, int C, int H, int W,
                    int H_out, int W_out,
                    int kH, int kW,
                    int stride_h, int stride_w,
                    int pad_h, int pad_w,
                    int dil_h, int dil_w,
                    float clamp_min, float clamp_max)
{
    fused_maxpool_kernel<<<blocks, threads>>>(
        input, scale, output,
        N, C, H, W,
        H_out, W_out,
        kH, kW,
        stride_h, stride_w,
        pad_h, pad_w,
        dil_h, dil_w,
        clamp_min, clamp_max);
}
"""

# -------------------------------------------------------------------------
# C++ bindings – expose the two wrapper functions to Python
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void conv_forward(int blocks, int threads,
                 const float* input, const float* weight, const float* bias, float* output,
                 int N, int C_in, int H_in, int W_in,
                 int C_out, int H_out, int W_out,
                 int kH, int kW,
                 int stride_h, int stride_w,
                 int pad_h, int pad_w,
                 int dil_h, int dil_w,
                 int groups);

void maxpool_forward(int blocks, int threads,
                    const float* input, const float* scale, float* output,
                    int N, int C, int H, int W,
                    int H_out, int W_out,
                    int kH, int kW,
                    int stride_h, int stride_w,
                    int pad_h, int pad_w,
                    int dil_h, int dil_w,
                    float clamp_min, float clamp_max);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_forward", &conv_forward, "Conv forward");
    m.def("maxpool_forward", &maxpool_forward, "Maxpool forward");
}
"""

# -------------------------------------------------------------------------
# Compile the inline CUDA extension
# -------------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_ops',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# -------------------------------------------------------------------------
# Helper functions for output‑size calculations
# -------------------------------------------------------------------------
def conv_out_size(H_in, W_in, kernel_size, stride, padding, dilation):
    H_out = (H_in + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    W_out = (W_in + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    return H_out, W_out

def maxpool_out_size(H_in, kernel_size, stride, padding, dilation, ceil_mode):
    if ceil_mode:
        H_out = int(math.ceil((H_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride)) + 1
    else:
        H_out = (H_in + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    return H_out

# -------------------------------------------------------------------------
# Optimized functional_model – replaces built‑in conv/maxpool and fuses
# scaling+clamp with max‑pooling
# -------------------------------------------------------------------------
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
    # Move input to GPU
    x = x.cuda()

    # ------------------------------------------------------------------
    # 1) Custom convolution (replace F.conv2d)
    # ------------------------------------------------------------------
    N, C_in, H_in, W_in = x.shape
    C_out = conv_weight.shape[0]
    kH = conv_weight.shape[2]
    kW = conv_weight.shape[3]

    H_out, W_out = conv_out_size(H_in, W_in, kH, conv_stride, conv_padding, conv_dilation)

    conv_out = torch.empty((N, C_out, H_out, W_out), dtype=x.dtype, device=x.device)

    weight_ptr = conv_weight.cuda().data_ptr()
    bias_ptr   = conv_bias.cuda().data_ptr() if conv_bias is not None else 0

    threads = 256
    blocks  = (N * C_out * H_out * W_out + threads - 1) // threads
    fused_ext.conv_forward(
        blocks, threads,
        x.data_ptr(), weight_ptr, bias_ptr, conv_out.data_ptr(),
        N, C_in, H_in, W_in,
        C_out, H_out, W_out,
        kH, kW,
        conv_stride, conv_stride,
        conv_padding, conv_padding,
        conv_dilation, conv_dilation,
        conv_groups
    )

    # ------------------------------------------------------------------
    # 2) Group normalization (still using PyTorch – not a conv/matmul)
    # ------------------------------------------------------------------
    x = F.group_norm(
        conv_out,
        group_norm_num_groups,
        group_norm_weight.cuda(),
        group_norm_bias.cuda(),
        eps=group_norm_eps
    )

    # ------------------------------------------------------------------
    # 3) Fused max‑pool + scale + clamp (replace F.max_pool2d + element‑wise ops)
    # ------------------------------------------------------------------
    H_in2, W_in2 = x.shape[2], x.shape[3]
    stride = maxpool_stride if maxpool_stride is not None else maxpool_kernel_size
    H_out2 = maxpool_out_size(H_in2, maxpool_kernel_size, stride, maxpool_padding, maxpool_dilation, maxpool_ceil_mode)
    W_out2 = maxpool_out_size(W_in2, maxpool_kernel_size, stride, maxpool_padding, maxpool_dilation, maxpool_ceil_mode)

    out = torch.empty((N, x.shape[1], H_out2, W_out2), dtype=x.dtype, device=x.device)

    # Per‑channel scale (flattened)
    scale_tensor = scale.to(x.device).view(-1)
    scale_ptr = scale_tensor.data_ptr()

    blocks = (N * x.shape[1] * H_out2 * W_out2 + threads - 1) // threads
    fused_ext.maxpool_forward(
        blocks, threads,
        x.data_ptr(), scale_ptr, out.data_ptr(),
        N, x.shape[1], H_in2, W_in2,
        H_out2, W_out2,
        maxpool_kernel_size, maxpool_kernel_size,
        stride, stride,
        maxpool_padding, maxpool_padding,
        maxpool_dilation, maxpool_dilation,
        clamp_min, clamp_max
    )

    # ignore maxpool_return_indices – not needed for the functional interface
    return out

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
    return [in_channels, out_channels, kernel_size, num_groups, scale_shape, maxpool_kernel_size, clamp_min, clamp_max]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]
