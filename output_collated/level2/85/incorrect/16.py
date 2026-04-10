# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_143406/code_3.py
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
import math

# ----------------------------------------------------------------------
# CUDA source – all kernels and the wrapper functions that PyTorch will call
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

// ----------------------------------------------------------------------
// 1. Grouped convolution kernel
// ----------------------------------------------------------------------
__global__ void conv_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const int N, const int C_in, const int H_in, const int W_in,
    const int C_out, const int kH, const int kW,
    const int stride_h, const int stride_w,
    const int pad_h, const int pad_w,
    const int dilation_h, const int dilation_w,
    const int groups,
    const int H_out, const int W_out,
    float* __restrict__ output)
{
    const int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_out = N * C_out * H_out * W_out;
    if (out_idx >= total_out) return;

    // decode (n, c, oh, ow)
    const int n = out_idx / (C_out * H_out * W_out);
    const int rem = out_idx % (C_out * H_out * W_out);
    const int c = rem / (H_out * W_out);
    const int rem2 = rem % (H_out * W_out);
    const int oh = rem2 / W_out;
    const int ow = rem2 % W_out;

    const int in_ch_per_group = C_in / groups;
    const int out_ch_per_group = C_out / groups;
    const int g = c / out_ch_per_group;          // group index

    float sum = 0.0f;

    // loop over input channels inside the group and the kernel
    for (int ic = 0; ic < in_ch_per_group; ++ic) {
        const int ic_full = g * in_ch_per_group + ic;
        for (int kh = 0; kh < kH; ++kh) {
            const int ih = oh * stride_h - pad_h + kh * dilation_h;
            if (ih < 0 || ih >= H_in) continue;
            for (int kw = 0; kw < kW; ++kw) {
                const int iw = ow * stride_w - pad_w + kw * dilation_w;
                if (iw < 0 || iw >= W_in) continue;
                const int in_idx = ((n * C_in + ic_full) * H_in + ih) * W_in + iw;
                const int w_idx = ((c * in_ch_per_group + ic) * kH + kh) * kW + kw;
                sum += input[in_idx] * weight[w_idx];
            }
        }
    }

    if (bias != nullptr) sum += bias[c];
    const int out_flat = ((n * C_out + c) * H_out + oh) * W_out + ow;
    output[out_flat] = sum;
}

void conv_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w,
    int groups,
    int H_out, int W_out,
    torch::Tensor output)
{
    const int N = input.size(0);
    const int C_in = input.size(1);
    const int H_in = input.size(2);
    const int W_in = input.size(3);
    const int C_out = weight.size(0);
    const int kH = weight.size(2);
    const int kW = weight.size(3);

    const int total_out = N * C_out * H_out * W_out;
    const int blocks = (total_out + 255) / 256;
    const int threads = 256;

    const float* bias_ptr = bias.numel() ? bias.data_ptr<float>() : nullptr;

    conv_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        N, C_in, H_in, W_in,
        C_out, kH, kW,
        stride_h, stride_w,
        pad_h, pad_w,
        dilation_h, dilation_w,
        groups,
        H_out, W_out,
        output.data_ptr<float>());
    cudaDeviceSynchronize();
}

// ----------------------------------------------------------------------
// 2. Kernel that computes per‑group sum and sum‑of‑squares
// ----------------------------------------------------------------------
__global__ void group_sum_kernel(
    const float* __restrict__ data,
    const int N, const int C, const int H, const int W,
    const int num_groups,
    float* __restrict__ sum,
    float* __restrict__ sum_sq)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * C * H * W;
    if (idx >= total) return;

    const int c = (idx / (H * W)) % C;
    const int group = c / (C / num_groups);
    const float val = data[idx];
    atomicAdd(&sum[group], val);
    atomicAdd(&sum_sq[group], val * val);
}

void compute_group_sum(
    torch::Tensor data,
    int N, int C, int H, int W,
    int num_groups,
    torch::Tensor sum,
    torch::Tensor sum_sq)
{
    const int total = N * C * H * W;
    const int blocks = (total + 255) / 256;
    const int threads = 256;
    group_sum_kernel<<<blocks, threads>>>(
        data.data_ptr<float>(),
        N, C, H, W,
        num_groups,
        sum.data_ptr<float>(),
        sum_sq.data_ptr<float>());
    cudaDeviceSynchronize();
}

// ----------------------------------------------------------------------
// 3. Kernel that normalises, adds weight/bias, multiplies by scale and clamps
// ----------------------------------------------------------------------
__global__ void groupnorm_scale_clamp_kernel(
    const float* __restrict__ data,
    const float* __restrict__ sum,
    const float* __restrict__ sum_sq,
    const int N, const int C, const int H, const int W,
    const int num_groups,
    const int group_size,
    const float* __restrict__ gn_weight,
    const float* __restrict__ gn_bias,
    const float eps,
    const float* __restrict__ scale,
    const float clamp_min,
    const float clamp_max,
    float* __restrict__ output)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * C * H * W;
    if (idx >= total) return;

    const int n = idx / (C * H * W);
    const int c = (idx / (H * W)) % C;
    const int group = c / (C / num_groups);

    const float mean = sum[group] / static_cast<float>(group_size);
    const float var = sum_sq[group] / static_cast<float>(group_size) - mean * mean;
    const float inv_std = rsqrtf(var + eps);

    float val = data[idx];
    val = (val - mean) * inv_std;
    val = val * gn_weight[c] + gn_bias[c];
    val = val * scale[c];
    if (val < clamp_min) val = clamp_min;
    else if (val > clamp_max) val = clamp_max;
    output[idx] = val;
}

void groupnorm_scale_clamp(
    torch::Tensor data,
    torch::Tensor sum,
    torch::Tensor sum_sq,
    int N, int C, int H, int W,
    int num_groups,
    int group_size,
    torch::Tensor gn_weight,
    torch::Tensor gn_bias,
    float eps,
    torch::Tensor scale,
    float clamp_min,
    float clamp_max,
    torch::Tensor output)
{
    const int total = N * C * H * W;
    const int blocks = (total + 255) / 256;
    const int threads = 256;
    groupnorm_scale_clamp_kernel<<<blocks, threads>>>(
        data.data_ptr<float>(),
        sum.data_ptr<float>(),
        sum_sq.data_ptr<float>(),
        N, C, H, W,
        num_groups,
        group_size,
        gn_weight.data_ptr<float>(),
        gn_bias.data_ptr<float>(),
        eps,
        scale.data_ptr<float>(),
        clamp_min, clamp_max,
        output.data_ptr<float>());
    cudaDeviceSynchronize();
}

// ----------------------------------------------------------------------
// 4. Kernel that performs max‑pooling and final clamp in one pass
// ----------------------------------------------------------------------
__global__ void maxpool_clamp_kernel(
    const float* __restrict__ input,
    const int N, const int C, const int H, const int W,
    const int kH, const int kW,
    const int stride_h, const int stride_w,
    const int pad_h, const int pad_w,
    const int dilation_h, const int dilation_w,
    const int H_out, const int W_out,
    const float clamp_min,
    const float clamp_max,
    float* __restrict__ output)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * C * H_out * W_out;
    if (idx >= total) return;

    const int n = idx / (C * H_out * W_out);
    const int rem = idx % (C * H_out * W_out);
    const int c = rem / (H_out * W_out);
    const int rem2 = rem % (H_out * W_out);
    const int oh = rem2 / W_out;
    const int ow = rem2 % W_out;

    float max_val = -INFINITY;
    for (int kh = 0; kh < kH; ++kh) {
        const int ih = oh * stride_h - pad_h + kh * dilation_h;
        if (ih < 0 || ih >= H) continue;
        for (int kw = 0; kw < kW; ++kw) {
            const int iw = ow * stride_w - pad_w + kw * dilation_w;
            if (iw < 0 || iw >= W) continue;
            const int in_idx = ((n * C + c) * H + ih) * W + iw;
            const float v = input[in_idx];
            if (v > max_val) max_val = v;
        }
    }
    float val = max_val;
    if (val < clamp_min) val = clamp_min;
    else if (val > clamp_max) val = clamp_max;
    output[idx] = val;
}

void maxpool_clamp(
    torch::Tensor input,
    int N, int C, int H, int W,
    int kH, int kW,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w,
    int H_out, int W_out,
    float clamp_min, float clamp_max,
    torch::Tensor output)
{
    const int total = N * C * H_out * W_out;
    const int blocks = (total + 255) / 256;
    const int threads = 256;
    maxpool_clamp_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        N, C, H, W,
        kH, kW,
        stride_h, stride_w,
        pad_h, pad_w,
        dilation_h, dilation_w,
        H_out, W_out,
        clamp_min, clamp_max,
        output.data_ptr<float>());
    cudaDeviceSynchronize();
}
"""

# ----------------------------------------------------------------------
# C++ bindings – expose the four functions to Python
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void conv_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w,
    int groups,
    int H_out, int W_out,
    torch::Tensor output);

void compute_group_sum(
    torch::Tensor data,
    int N, int C, int H, int W,
    int num_groups,
    torch::Tensor sum,
    torch::Tensor sum_sq);

void groupnorm_scale_clamp(
    torch::Tensor data,
    torch::Tensor sum,
    torch::Tensor sum_sq,
    int N, int C, int H, int W,
    int num_groups,
    int group_size,
    torch::Tensor gn_weight,
    torch::Tensor gn_bias,
    float eps,
    torch::Tensor scale,
    float clamp_min,
    float clamp_max,
    torch::Tensor output);

void maxpool_clamp(
    torch::Tensor input,
    int N, int C, int H, int W,
    int kH, int kW,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w,
    int H_out, int W_out,
    float clamp_min, float clamp_max,
    torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_forward", &conv_forward, "grouped convolution forward");
    m.def("compute_group_sum", &compute_group_sum, "group sum for GroupNorm");
    m.def("groupnorm_scale_clamp", &groupnorm_scale_clamp, "group norm + scale + clamp");
    m.def("maxpool_clamp", &maxpool_clamp, "max pooling + clamp");
}
"""

# ----------------------------------------------------------------------
# Compile the extension
# ----------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True)

# ----------------------------------------------------------------------
# Helper to compute spatial output size of a convolution or pooling
# ----------------------------------------------------------------------
def out_size(in_size, kernel, stride, pad, dilation, ceil_mode):
    numerator = in_size + 2 * pad - dilation * (kernel - 1) - 1
    if ceil_mode:
        return (numerator + stride - 1) // stride + 1
    else:
        return numerator // stride + 1

# ----------------------------------------------------------------------
# The functional model that will be imported and evaluated
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
    # Ensure all inputs are on the GPU
    # ------------------------------------------------------------------
    x = x.cuda()
    conv_weight = conv_weight.cuda()
    if conv_bias is not None:
        conv_bias = conv_bias.cuda()
    else:
        # create an empty tensor that will be interpreted as “no bias” in the kernel
        conv_bias = torch.tensor([], dtype=torch.float32, device='cuda')
    group_norm_weight = group_norm_weight.cuda()
    group_norm_bias = group_norm_bias.cuda()
    scale = scale.cuda()

    # ------------------------------------------------------------------
    # Unpack / normalise stride, padding, dilation (they may be int or tuple)
    # ------------------------------------------------------------------
    if isinstance(conv_stride, int):
        stride_h = stride_w = conv_stride
    else:
        stride_h, stride_w = conv_stride

    if isinstance(conv_padding, int):
        pad_h = pad_w = conv_padding
    else:
        pad_h, pad_w = conv_padding

    if isinstance(conv_dilation, int):
        dilation_h = dilation_w = conv_dilation
    else:
        dilation_h, dilation_w = conv_dilation

    # ------------------------------------------------------------------
    # 1) Convolution
    # ------------------------------------------------------------------
    N, C_in, H_in, W_in = x.shape
    C_out = conv_weight.shape[0]
    kH = conv_weight.shape[2]
    kW = conv_weight.shape[3]

    H_out_conv = out_size(H_in, kH, stride_h, pad_h, dilation_h, False)
    W_out_conv = out_size(W_in, kW, stride_w, pad_w, dilation_w, False)

    conv_out = torch.empty(N, C_out, H_out_conv, W_out_conv,
                           dtype=torch.float32, device='cuda')

    fused_ext.conv_forward(
        x, conv_weight, conv_bias,
        stride_h, stride_w,
        pad_h, pad_w,
        dilation_h, dilation_w,
        conv_groups,
        H_out_conv, W_out_conv,
        conv_out)

    # ------------------------------------------------------------------
    # 2) Group‑norm (two‑pass) + scale + clamp
    # ------------------------------------------------------------------
    num_groups = group_norm_num_groups
    sum_buf = torch.zeros(num_groups, dtype=torch.float32, device='cuda')
    sum_sq_buf = torch.zeros(num_groups, dtype=torch.float32, device='cuda')

    fused_ext.compute_group_sum(
        conv_out,
        N, C_out, H_out_conv, W_out_conv,
        num_groups,
        sum_buf, sum_sq_buf)

    group_size = N * (C_out // num_groups) * H_out_conv * W_out_conv

    norm_out = torch.empty(N, C_out, H_out_conv, W_out_conv,
                           dtype=torch.float32, device='cuda')

    fused_ext.groupnorm_scale_clamp(
        conv_out,
        sum_buf, sum_sq_buf,
        N, C_out, H_out_conv, W_out_conv,
        num_groups,
        group_size,
        group_norm_weight, group_norm_bias,
        group_norm_eps,
        scale,
        clamp_min, clamp_max,
        norm_out)

    # ------------------------------------------------------------------
    # 3) Max‑pooling + final clamp
    # ------------------------------------------------------------------
    if isinstance(maxpool_stride, int):
        mp_stride_h = mp_stride_w = maxpool_stride
    else:
        mp_stride_h, mp_stride_w = maxpool_stride

    if isinstance(maxpool_padding, int):
        mp_pad_h = mp_pad_w = maxpool_padding
    else:
        mp_pad_h, mp_pad_w = maxpool_padding

    if isinstance(maxpool_dilation, int):
        mp_dilation_h = mp_dilation_w = maxpool_dilation
    else:
        mp_dilation_h, mp_dilation_w = maxpool_dilation

    H_out_mp = out_size(H_out_conv, maxpool_kernel_size,
                        mp_stride_h, mp_pad_h, mp_dilation_h,
                        maxpool_ceil_mode)
    W_out_mp = out_size(W_out_conv, maxpool_kernel_size,
                        mp_stride_w, mp_pad_w, mp_dilation_w,
                        maxpool_ceil_mode)

    mp_out = torch.empty(N, C_out, H_out_mp, W_out_mp,
                         dtype=torch.float32, device='cuda')

    fused_ext.maxpool_clamp(
        norm_out,
        N, C_out, H_out_conv, W_out_conv,
        maxpool_kernel_size, maxpool_kernel_size,
        mp_stride_h, mp_stride_w,
        mp_pad_h, mp_pad_w,
        mp_dilation_h, mp_dilation_w,
        H_out_mp, W_out_mp,
        clamp_min, clamp_max,
        mp_out)

    # ------------------------------------------------------------------
    # Return the final tensor – note that we ignore `maxpool_return_indices`
    # because the original code returns the pooled tensor, not the indices.
    # ------------------------------------------------------------------
    return mp_out

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
