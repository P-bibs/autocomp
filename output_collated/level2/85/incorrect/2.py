# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_141453/code_3.py
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
import math
from torch.utils.cpp_extension import load_inline

# -------------------------------------------------------------------------
# 1.  CUDA source (kernels + wrappers)
# -------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

// ---------------------------------------------------------------------
// 1.1  Convolution – naive but sufficient for the given sizes
// ---------------------------------------------------------------------
__global__ void conv_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int N, const int C_in, const int H_in, const int W_in,
    const int C_out, const int K, const int stride, const int padding,
    const int dilation, const int groups,
    const int H_out, const int W_out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * C_out * H_out * W_out;
    if (idx >= total) return;

    int w_out = idx % W_out;
    int h_out = (idx / W_out) % H_out;
    int c_out = (idx / (W_out * H_out)) % C_out;
    int n    = idx / (W_out * H_out * C_out);

    float sum = 0.0f;
    if (bias) sum += bias[c_out];

    const int c_in_per_group = C_in / groups;
    const int group_id       = c_out / (C_out / groups);

    for (int ci = 0; ci < c_in_per_group; ++ci) {
        int c_in = group_id * c_in_per_group + ci;
        for (int kh = 0; kh < K; ++kh) {
            int h_in = h_out * stride - padding + kh * dilation;
            if (h_in < 0 || h_in >= H_in) continue;
            for (int kw = 0; kw < K; ++kw) {
                int w_in = w_out * stride - padding + kw * dilation;
                if (w_in < 0 || w_in >= W_in) continue;
                int in_idx  = ((n * C_in + c_in) * H_in + h_in) * W_in + w_in;
                int w_idx   = ((c_out * c_in_per_group + ci) * K + kh) * K + kw;
                sum += input[in_idx] * weight[w_idx];
            }
        }
    }
    output[((n * C_out + c_out) * H_out + h_out) * W_out + w_out] = sum;
}

// ---------------------------------------------------------------------
// 1.2  Compute per‑group sum and sum‑of‑squares (for GroupNorm)
// ---------------------------------------------------------------------
__global__ void compute_group_sum_kernel(
    const float* __restrict__ conv_out,
    float* __restrict__ group_sum,
    float* __restrict__ group_sum_sq,
    const int N, const int C_out, const int H_out, const int W_out,
    const int num_groups)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * C_out * H_out * W_out;
    if (idx >= total) return;

    int w = idx % W_out;
    int h = (idx / W_out) % H_out;
    int c = (idx / (W_out * H_out)) % C_out;
    int n = idx / (W_out * H_out * C_out);

    int group_id = c / (C_out / num_groups);   // C_out is guaranteed divisible by num_groups
    float val = conv_out[idx];
    atomicAdd(&group_sum[group_id], val);
    atomicAdd(&group_sum_sq[group_id], val * val);
}

// ---------------------------------------------------------------------
// 1.3  Fused kernel: GroupNorm → scale → max‑pool → clamp
// ---------------------------------------------------------------------
__global__ void fused_norm_scale_pool_clamp_kernel(
    const float* __restrict__ conv_out,
    const float* __restrict__ group_sum,
    const float* __restrict__ group_sum_sq,
    const float* __restrict__ group_weight,
    const float* __restrict__ group_bias,
    const float* __restrict__ scale,          // per‑channel scale (size C_out)
    float* __restrict__ output,
    const int N, const int C_out, const int H_out, const int W_out,
    const int num_groups,
    const float eps,
    const int pool_k,
    const int pool_stride,
    const int pool_padding,
    const int pool_dilation,
    const int pool_ceil_mode,
    const float clamp_min,
    const float clamp_max,
    const int H_pool,
    const int W_pool)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * C_out * H_pool * W_pool;
    if (idx >= total) return;

    int w = idx % W_pool;
    int h = (idx / W_pool) % H_pool;
    int c = (idx / (W_pool * H_pool)) % C_out;
    int n = idx / (W_pool * H_pool * C_out);

    // location of the pooling window in the input (conv_out) tensor
    int h_start = h * pool_stride - pool_padding;
    int w_start = w * pool_stride - pool_padding;

    // Group‑Norm statistics for this channel
    const int group_id   = c / (C_out / num_groups);
    const float sum      = group_sum[group_id];
    const float sum_sq   = group_sum_sq[group_id];
    const float count    = static_cast<float>(N * H_out * W_out);
    const float mean     = sum / count;
    const float var      = sum_sq / count - mean * mean;
    const float inv_std  = rsqrtf(var + eps);

    // pooling (max)
    float max_val = -1e38f;
    #pragma unroll 4
    for (int ki = 0; ki < pool_k; ++ki) {
        int h_in = h_start + ki * pool_dilation;
        #pragma unroll 4
        for (int kj = 0; kj < pool_k; ++kj) {
            int w_in = w_start + kj * pool_dilation;
            if (h_in >= 0 && h_in < H_out && w_in >= 0 && w_in < W_out) {
                int idx_in = ((n * C_out + c) * H_out + h_in) * W_out + w_in;
                float val = conv_out[idx_in];
                // GroupNorm
                float norm = (val - mean) * inv_std;
                // per‑channel weight & bias
                float out = norm * group_weight[c] + group_bias[c];
                // per‑channel scale
                out *= scale[c];
                // keep max
                if (out > max_val) max_val = out;
            }
        }
    }

    // clamp
    if (max_val < clamp_min) max_val = clamp_min;
    else if (max_val > clamp_max) max_val = clamp_max;

    output[idx] = max_val;
}

// ---------------------------------------------------------------------
// 1.4  C++ wrappers (PYBIND11)
// ---------------------------------------------------------------------
void conv_forward(
    const float* input, const float* weight, const float* bias,
    float* output,
    int N, int C_in, int H_in, int W_in,
    int C_out, int K, int stride, int padding, int dilation, int groups,
    int H_out, int W_out)
{
    int threads = 256;
    int blocks  = (N * C_out * H_out * W_out + threads - 1) / threads;
    conv_forward_kernel<<<blocks, threads>>>(
        input, weight, bias, output,
        N, C_in, H_in, W_in,
        C_out, K, stride, padding, dilation, groups,
        H_out, W_out);
    cudaDeviceSynchronize();
}

void compute_group_sum(
    const float* conv_out,
    float* group_sum, float* group_sum_sq,
    int N, int C_out, int H_out, int W_out,
    int num_groups)
{
    int threads = 256;
    int total   = N * C_out * H_out * W_out;
    int blocks  = (total + threads - 1) / threads;
    compute_group_sum_kernel<<<blocks, threads>>>(
        conv_out, group_sum, group_sum_sq,
        N, C_out, H_out, W_out, num_groups);
    cudaDeviceSynchronize();
}

void fused_norm_scale_pool_clamp(
    const float* conv_out,
    const float* group_sum,
    const float* group_sum_sq,
    const float* group_weight,
    const float* group_bias,
    const float* scale,
    float* output,
    int N, int C_out, int H_out, int W_out,
    int num_groups,
    float eps,
    int pool_k, int pool_stride, int pool_padding, int pool_dilation,
    int pool_ceil_mode,
    float clamp_min, float clamp_max,
    int H_pool, int W_pool)
{
    int threads = 256;
    int total   = N * C_out * H_pool * W_pool;
    int blocks  = (total + threads - 1) / threads;
    fused_norm_scale_pool_clamp_kernel<<<blocks, threads>>>(
        conv_out, group_sum, group_sum_sq,
        group_weight, group_bias, scale,
        output,
        N, C_out, H_out, W_out,
        num_groups, eps,
        pool_k, pool_stride, pool_padding, pool_dilation,
        pool_ceil_mode, clamp_min, clamp_max,
        H_pool, W_pool);
    cudaDeviceSynchronize();
}
"""

# -------------------------------------------------------------------------
# 2.  C++ binding (PYBIND11)
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void conv_forward(
    const float* input, const float* weight, const float* bias,
    float* output,
    int N, int C_in, int H_in, int W_in,
    int C_out, int K, int stride, int padding, int dilation, int groups,
    int H_out, int W_out);

void compute_group_sum(
    const float* conv_out,
    float* group_sum, float* group_sum_sq,
    int N, int C_out, int H_out, int W_out,
    int num_groups);

void fused_norm_scale_pool_clamp(
    const float* conv_out,
    const float* group_sum,
    const float* group_sum_sq,
    const float* group_weight,
    const float* group_bias,
    const float* scale,
    float* output,
    int N, int C_out, int H_out, int W_out,
    int num_groups,
    float eps,
    int pool_k, int pool_stride, int pool_padding, int pool_dilation,
    int pool_ceil_mode,
    float clamp_min, float clamp_max,
    int H_pool, int W_pool);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_forward", &conv_forward, "conv forward");
    m.def("compute_group_sum", &compute_group_sum, "compute group sums");
    m.def("fused_norm_scale_pool_clamp", &fused_norm_scale_pool_clamp,
          "fused norm+scale+maxpool+clamp");
}
"""

# -------------------------------------------------------------------------
# 3.  Compile the extension
# -------------------------------------------------------------------------
fused_ext = load_inline(
    name="fused_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    with_cuda=True,
)

# -------------------------------------------------------------------------
# 4.  Helper functions (kept for reference / may be used by the harness)
# -------------------------------------------------------------------------
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
    return [in_channels, out_channels, kernel_size, num_groups, scale_shape,
            maxpool_kernel_size, clamp_min, clamp_max]


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


# -------------------------------------------------------------------------
# 5.  The optimized functional_model (fused pipeline)
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
    # -----------------------------------------------------------------
    # Move everything to GPU if not already there
    # -----------------------------------------------------------------
    if not x.is_cuda:
        x = x.cuda()
    if not conv_weight.is_cuda:
        conv_weight = conv_weight.cuda()
    if conv_bias is not None and not conv_bias.is_cuda:
        conv_bias = conv_bias.cuda()
    if not group_norm_weight.is_cuda:
        group_norm_weight = group_norm_weight.cuda()
    if not group_norm_bias.is_cuda:
        group_norm_bias = group_norm_bias.cuda()

    # -----------------------------------------------------------------
    # 5.1  Prepare data (ensure contiguity, dummy bias if needed)
    # -----------------------------------------------------------------
    N = x.size(0)
    C_in = x.size(1)
    H_in = x.size(2)
    W_in = x.size(3)

    out_ch = conv_weight.size(0)
    K = conv_weight.size(2)

    # If bias not supplied, use a zero tensor (the kernel always adds it)
    if conv_bias is None:
        conv_bias = torch.zeros(out_ch, dtype=torch.float32, device="cuda")

    # -----------------------------------------------------------------
    # 5.2  Compute output size after convolution
    # -----------------------------------------------------------------
    H_out = (H_in + 2 * conv_padding - conv_dilation * (K - 1) - 1) // conv_stride + 1
    W_out = (W_in + 2 * conv_padding - conv_dilation * (K - 1) - 1) // conv_stride + 1

    # Allocate intermediate tensor
    conv_out = torch.empty((N, out_ch, H_out, W_out),
                           dtype=torch.float32, device="cuda")

    # -----------------------------------------------------------------
    # 5.3  Launch custom convolution kernel
    # -----------------------------------------------------------------
    fused_ext.conv_forward(
        x.contiguous().data_ptr(),
        conv_weight.contiguous().data_ptr(),
        conv_bias.contiguous().data_ptr(),
        conv_out.data_ptr(),
        N, C_in, H_in, W_in,
        out_ch, K, conv_stride, conv_padding, conv_dilation, conv_groups,
        H_out, W_out)

    # -----------------------------------------------------------------
    # 5.4  Group‑Norm statistics (one pass to compute sums)
    # -----------------------------------------------------------------
    num_groups = group_norm_num_groups
    group_sum = torch.zeros(num_groups, dtype=torch.float32, device="cuda")
    group_sum_sq = torch.zeros(num_groups, dtype=torch.float32, device="cuda")

    fused_ext.compute_group_sum(
        conv_out.data_ptr(),
        group_sum.data_ptr(),
        group_sum_sq.data_ptr(),
        N, out_ch, H_out, W_out, num_groups)

    # -----------------------------------------------------------------
    # 5.5  Compute output size after max‑pooling
    # -----------------------------------------------------------------
    if maxpool_ceil_mode:
        H_pool = (H_out + 2 * maxpool_padding - maxpool_dilation * (maxpool_kernel_size - 1) + maxpool_stride - 1) // maxpool_stride
        W_pool = (W_out + 2 * maxpool_padding - maxpool_dilation * (maxpool_kernel_size - 1) + maxpool_stride - 1) // maxpool_stride
    else:
        H_pool = (H_out + 2 * maxpool_padding - maxpool_dilation * (maxpool_kernel_size - 1) - 1) // maxpool_stride + 1
        W_pool = (W_out + 2 * maxpool_padding - maxpool_dilation * (maxpool_kernel_size - 1) - 1) // maxpool_stride + 1

    output = torch.empty((N, out_ch, H_pool, W_pool),
                         dtype=torch.float32, device="cuda")

    # -----------------------------------------------------------------
    # 5.6  Prepare per‑channel scale tensor
    # -----------------------------------------------------------------
    # scale can be a scalar or a tensor of shape (out_ch,1,1).  Flatten it.
    if isinstance(scale, torch.Tensor):
        scale_tensor = scale.reshape(out_ch).contiguous()
    else:
        scale_tensor = torch.full((out_ch,), scale,
                                  dtype=torch.float32, device="cuda")

    # -----------------------------------------------------------------
    # 5.7  Launch fused GroupNorm + scale + max‑pool + clamp kernel
    # -----------------------------------------------------------------
    fused_ext.fused_norm_scale_pool_clamp(
        conv_out.data_ptr(),
        group_sum.data_ptr(),
        group_sum_sq.data_ptr(),
        group_norm_weight.contiguous().data_ptr(),
        group_norm_bias.contiguous().data_ptr(),
        scale_tensor.data_ptr(),
        output.data_ptr(),
        N, out_ch, H_out, W_out,
        num_groups,
        group_norm_eps,
        maxpool_kernel_size,
        maxpool_stride,
        maxpool_padding,
        maxpool_dilation,
        1 if maxpool_ceil_mode else 0,
        clamp_min,
        clamp_max,
        H_pool,
        W_pool)

    return output
