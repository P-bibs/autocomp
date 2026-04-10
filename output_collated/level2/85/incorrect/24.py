# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_144231/code_3.py
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# ----------------------------------------------------------------------
# Helper: convert possibly‑tuple arguments to a single int
# ----------------------------------------------------------------------
def _to_int(x):
    if isinstance(x, (int, float)):
        return int(x)
    elif isinstance(x, (tuple, list)):
        return int(x[0])
    else:
        return int(x)

# ----------------------------------------------------------------------
# Inline CUDA / C++ code – three custom kernels + bindings
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

// ----------------------------------------------------------------------
// 1. Naïve convolution kernel (supports stride, padding, dilation, groups)
// ----------------------------------------------------------------------
__global__ void conv_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* output,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int kernel_size, int stride, int padding, int dilation,
    int groups) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_out * H_out * W_out;
    if (idx >= total) return;

    int w_out = idx % W_out;
    int tmp = idx / W_out;
    int h_out = tmp % H_out;
    int c_out_tmp = tmp / H_out;
    int n = c_out_tmp / C_out;
    int c_out = c_out_tmp % C_out;

    float sum = 0.0f;
    int h_start = h_out * stride - padding;
    int w_start = w_out * stride - padding;

    // input‑channel range for the current group
    int groups_in  = C_in  / groups;
    int groups_out = C_out / groups;
    int g = c_out / groups_out;
    int c_in_start = g * groups_in;
    int c_in_end   = c_in_start + groups_in;

    for (int c_in = c_in_start; c_in < c_in_end; ++c_in) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            int h_in = h_start + kh * dilation;
            if (h_in < 0 || h_in >= H_in) continue;
            for (int kw = 0; kw < kernel_size; ++kw) {
                int w_in = w_start + kw * dilation;
                if (w_in < 0 || w_in >= W_in) continue;
                float w_val = weight[((c_out * C_in + c_in) * kernel_size + kh) * kernel_size + kw];
                float i_val = input[((n * C_in + c_in) * H_in + h_in) * W_in + w_in];
                sum += w_val * i_val;
            }
        }
    }

    if (bias != nullptr) sum += bias[c_out];
    output[idx] = sum;
}

void conv_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int kernel_size, int stride, int padding, int dilation,
    int groups) {

    const float* in_ptr  = input.data_ptr<float>();
    const float* w_ptr   = weight.data_ptr<float>();
    const float* b_ptr   = nullptr;
    if (bias.defined() && bias.numel() > 0) b_ptr = bias.data_ptr<float>();
    float* out_ptr = output.data_ptr<float>();

    int total = N * C_out * H_out * W_out;
    int blocks = (total + 255) / 256;
    conv_kernel<<<blocks, 256>>>(
        in_ptr, w_ptr, b_ptr, out_ptr,
        N, C_in, H_in, W_in, C_out, H_out, W_out,
        kernel_size, stride, padding, dilation, groups);
    cudaDeviceSynchronize();
}

// ----------------------------------------------------------------------
// 2. Fused Group‑Norm + Scale kernel
// ----------------------------------------------------------------------
__global__ void group_norm_scale_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,   // group_norm weight (C_out)
    const float* __restrict__ bias,     // group_norm bias   (C_out)
    const float* __restrict__ scale,    // per‑channel scale (C_out)
    float* output,
    int N, int C, int H, int W,
    int num_groups, float eps) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;
    if (idx >= total) return;

    int w = idx % W;
    int tmp = idx / W;
    int h = tmp % H;
    int c_tmp = tmp / H;
    int n = c_tmp / C;
    int c = c_tmp % C;

    int group_size = C / num_groups;
    int g = c / group_size;
    int group_start = g * group_size;

    float sum = 0.0f, sum_sq = 0.0f;
    for (int cc = group_start; cc < group_start + group_size; ++cc) {
        float v = input[((n * C + cc) * H + h) * W + w];
        sum   += v;
        sum_sq += v * v;
    }
    float mean   = sum / (float)group_size;
    float var    = sum_sq / (float)group_size - mean * mean;
    if (var < 0) var = 0;
    float inv_std = rsqrtf(var + eps);

    float v = input[idx];
    float normalized = (v - mean) * inv_std * weight[c] + bias[c];
    float scaled = normalized * scale[c];
    output[idx] = scaled;
}

void group_norm_scale_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor scale,
    torch::Tensor output,
    int N, int C, int H, int W,
    int num_groups,
    float eps) {

    const float* in_ptr  = input.data_ptr<float>();
    const float* w_ptr   = weight.data_ptr<float>();
    const float* b_ptr   = bias.data_ptr<float>();
    const float* s_ptr   = scale.data_ptr<float>();
    float* out_ptr = output.data_ptr<float>();

    int total = N * C * H * W;
    int blocks = (total + 255) / 256;
    group_norm_scale_kernel<<<blocks, 256>>>(
        in_ptr, w_ptr, b_ptr, s_ptr, out_ptr,
        N, C, H, W, num_groups, eps);
    cudaDeviceSynchronize();
}

// ----------------------------------------------------------------------
// 3. Fused Max‑Pool + Clamp kernel
// ----------------------------------------------------------------------
__global__ void maxpool_clamp_kernel(
    const float* __restrict__ input,
    float* output,
    int N, int C, int H_in, int W_in,
    int kernel_size, int stride, int padding,
    float clamp_min, float clamp_max) {

    int out_h = (H_in - kernel_size) / stride + 1;
    int out_w = (W_in - kernel_size) / stride + 1;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * out_h * out_w;
    if (idx >= total) return;

    int w_out = idx % out_w;
    int tmp = idx / out_w;
    int h_out = tmp % out_h;
    int c_tmp = tmp / out_h;
    int n = c_tmp / C;
    int c = c_tmp % C;

    int h_start = h_out * stride - padding;
    int w_start = w_out * stride - padding;

    float max_val = -FLT_MAX;
    for (int kh = 0; kh < kernel_size; ++kh) {
        int h = h_start + kh;
        if (h < 0 || h >= H_in) continue;
        for (int kw = 0; kw < kernel_size; ++kw) {
            int w = w_start + kw;
            if (w < 0 || w >= W_in) continue;
            float v = input[((n * C + c) * H_in + h) * W_in + w];
            if (v > max_val) max_val = v;
        }
    }
    float v = fmaxf(fminf(max_val, clamp_max), clamp_min);
    output[idx] = v;
}

void maxpool_clamp_forward(
    torch::Tensor input,
    torch::Tensor output,
    int N, int C, int H_in, int W_in,
    int kernel_size, int stride, int padding,
    float clamp_min, float clamp_max) {

    const float* in_ptr  = input.data_ptr<float>();
    float* out_ptr = output.data_ptr<float>();

    int out_h = (H_in - kernel_size) / stride + 1;
    int out_w = (W_in - kernel_size) / stride + 1;
    int total = N * C * out_h * out_w;
    int blocks = (total + 255) / 256;
    maxpool_clamp_kernel<<<blocks, 256>>>(
        in_ptr, out_ptr,
        N, C, H_in, W_in,
        kernel_size, stride, padding,
        clamp_min, clamp_max);
    cudaDeviceSynchronize();
}
"""

cpp_source = r"""
#include <torch/extension.h>

void conv_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int kernel_size, int stride, int padding, int dilation,
    int groups);

void group_norm_scale_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor scale,
    torch::Tensor output,
    int N, int C, int H, int W,
    int num_groups,
    float eps);

void maxpool_clamp_forward(
    torch::Tensor input,
    torch::Tensor output,
    int N, int C, int H_in, int W_in,
    int kernel_size, int stride, int padding,
    float clamp_min, float clamp_max);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_forward", &conv_forward, "Convolution forward kernel");
    m.def("group_norm_scale_forward", &group_norm_scale_forward, "GroupNorm + Scale forward kernel");
    m.def("maxpool_clamp_forward", &maxpool_clamp_forward, "MaxPool + Clamp forward kernel");
}
"""

# Compile the fused extension
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# ----------------------------------------------------------------------
# Original helper functions (kept for completeness)
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

# ----------------------------------------------------------------------
# Optimised functional_model – uses the three custom kernels
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
    # Move everything to GPU
    # ------------------------------------------------------------------
    x = x.cuda()
    conv_weight = conv_weight.cuda()
    if conv_bias is not None:
        conv_bias = conv_bias.cuda()
    group_norm_weight = group_norm_weight.cuda()
    group_norm_bias   = group_norm_bias.cuda()
    scale             = scale.cuda()

    # ------------------------------------------------------------------
    # Convert possibly‑tuple parameters to plain ints
    # ------------------------------------------------------------------
    stride   = _to_int(conv_stride)
    padding  = _to_int(conv_padding)
    dilation = _to_int(conv_dilation)
    groups   = int(conv_groups)

    maxpool_stride_val = maxpool_kernel_size if maxpool_stride is None else _to_int(maxpool_stride)
    maxpool_padding_val = 0 if maxpool_padding is None else _to_int(maxpool_padding)

    # ------------------------------------------------------------------
    # 1) Custom convolution
    # ------------------------------------------------------------------
    N, C_in, H_in, W_in = x.shape
    C_out = conv_weight.size(0)
    ksz   = conv_weight.size(2)

    H_out = (H_in + 2 * padding - dilation * (ksz - 1) - 1) // stride + 1
    W_out = (W_in + 2 * padding - dilation * (ksz - 1) - 1) // stride + 1

    conv_out = torch.empty((N, C_out, H_out, W_out), dtype=torch.float32, device='cuda')

    fused_ext.conv_forward(
        x, conv_weight,
        conv_bias if conv_bias is not None else torch.empty(0, device='cuda'),
        conv_out,
        N, C_in, H_in, W_in,
        C_out, H_out, W_out,
        ksz, stride, padding, dilation, groups)

    # ------------------------------------------------------------------
    # 2) Fused Group‑Norm + Scale
    # ------------------------------------------------------------------
    conv_norm = torch.empty_like(conv_out)
    fused_ext.group_norm_scale_forward(
        conv_out,
        group_norm_weight,
        group_norm_bias,
        scale,
        conv_norm,
        N, C_out, H_out, W_out,
        group_norm_num_groups,
        group_norm_eps)

    # ------------------------------------------------------------------
    # 3) Fused Max‑Pool + Clamp
    # ------------------------------------------------------------------
    H_max_out = (H_out - maxpool_kernel_size) // maxpool_stride_val + 1
    W_max_out = (W_out - maxpool_kernel_size) // maxpool_stride_val + 1

    output = torch.empty((N, C_out, H_max_out, W_max_out),
                         dtype=torch.float32, device='cuda')

    fused_ext.maxpool_clamp_forward(
        conv_norm,
        output,
        N, C_out, H_out, W_out,
        maxpool_kernel_size, maxpool_stride_val, maxpool_padding_val,
        clamp_min, clamp_max)

    return output
