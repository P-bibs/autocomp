# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_060810/code_7.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'divisor', 'pool_size', 'bias_shape', 'sum_dim']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_weight', 'conv_bias', 'conv_stride', 'conv_padding', 'conv_dilation', 'conv_groups', 'max_pool_kernel_size', 'max_pool_stride', 'max_pool_padding', 'max_pool_dilation', 'max_pool_ceil_mode', 'max_pool_return_indices', 'global_avg_pool_output_size', 'divisor', 'bias', 'sum_dim']
REQUIRED_FLAT_STATE_NAMES = ['conv_weight', 'conv_bias', 'bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a 3D convolution, divides by a constant, applies max pooling,
    global average pooling, adds a bias term, and sums along a specific dimension.
    """

    def __init__(self, in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.divisor = divisor
        self.max_pool = nn.MaxPool3d(pool_size)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.sum_dim = sum_dim

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
    # State for conv (nn.Conv3d)
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
    # State for max_pool (nn.MaxPool3d)
    state_kwargs['max_pool_kernel_size'] = model.max_pool.kernel_size
    state_kwargs['max_pool_stride'] = model.max_pool.stride
    state_kwargs['max_pool_padding'] = model.max_pool.padding
    state_kwargs['max_pool_dilation'] = model.max_pool.dilation
    state_kwargs['max_pool_ceil_mode'] = model.max_pool.ceil_mode
    state_kwargs['max_pool_return_indices'] = model.max_pool.return_indices
    # State for global_avg_pool (nn.AdaptiveAvgPool3d)
    state_kwargs['global_avg_pool_output_size'] = model.global_avg_pool.output_size
    if 'divisor' in flat_state:
        state_kwargs['divisor'] = flat_state['divisor']
    else:
        state_kwargs['divisor'] = getattr(model, 'divisor')
    if 'bias' in flat_state:
        state_kwargs['bias'] = flat_state['bias']
    else:
        state_kwargs['bias'] = getattr(model, 'bias')
    if 'sum_dim' in flat_state:
        state_kwargs['sum_dim'] = flat_state['sum_dim']
    else:
        state_kwargs['sum_dim'] = getattr(model, 'sum_dim')
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
# CUDA source – three kernels + wrapper functions that launch them
# ----------------------------------------------------------------------
cuda_src = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// ----------------------------------------------------------------------
// 1) 3‑D convolution (naïve outer‑product) – also does the division
// ----------------------------------------------------------------------
__global__ void conv3d_kernel(const float* __restrict__ input,
                              const float* __restrict__ weight,
                              float* __restrict__ output,
                              const int N, const int C_in, const int D_in, const int H_in, const int W_in,
                              const int C_out, const int Kd, const int Kh, const int Kw,
                              const int s0, const int s1, const int s2,
                              const int p0, const int p1, const int p2,
                              const int d0, const int d1, const int d2,
                              const int D_out, const int H_out, const int W_out,
                              const float divisor) {
    const int total = N * C_out * D_out * H_out * W_out;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int tmp = idx;
    int n = tmp / (C_out * D_out * H_out * W_out);
    tmp %= (C_out * D_out * H_out * W_out);
    int c = tmp / (D_out * H_out * W_out);
    tmp %= (D_out * H_out * W_out);
    int d = tmp / (H_out * W_out);
    tmp %= (H_out * W_out);
    int h = tmp / W_out;
    int w = tmp % W_out;

    float sum = 0.0f;
    for (int ic = 0; ic < C_in; ++ic) {
        for (int kd = 0; kd < Kd; ++kd) {
            int i_d = d * s0 - p0 + kd * d0;
            if (i_d < 0 || i_d >= D_in) continue;
            for (int kh = 0; kh < Kh; ++kh) {
                int i_h = h * s1 - p1 + kh * d1;
                if (i_h < 0 || i_h >= H_in) continue;
                for (int kw = 0; kw < Kw; ++kw) {
                    int i_w = w * s2 - p2 + kw * d2;
                    if (i_w < 0 || i_w >= W_in) continue;
                    int w_idx = (((c * C_in + ic) * Kd + kd) * Kh + kh) * Kw + kw;
                    int i_idx = (((n * C_in + ic) * D_in + i_d) * H_in + i_h) * W_in + i_w;
                    sum += weight[w_idx] * input[i_idx];
                }
            }
        }
    }
    sum /= divisor;
    int o_idx = (((n * C_out + c) * D_out + d) * H_out + h) * W_out + w;
    output[o_idx] = sum;
}

// ----------------------------------------------------------------------
// 2) 3‑D max‑pooling (naïve sliding window)
// ----------------------------------------------------------------------
__global__ void maxpool3d_kernel(const float* __restrict__ input,
                                 float* __restrict__ output,
                                 const int N, const int C,
                                 const int D_in, const int H_in, const int W_in,
                                 const int D_out, const int H_out, const int W_out,
                                 const int kd, const int kh, const int kw,
                                 const int s0, const int s1, const int s2,
                                 const int p0, const int p1, const int p2,
                                 const int d0, const int d1, const int d2) {
    const int total = N * C * D_out * H_out * W_out;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int tmp = idx;
    int n = tmp / (C * D_out * H_out * W_out);
    tmp %= (C * D_out * H_out * W_out);
    int c = tmp / (D_out * H_out * W_out);
    tmp %= (D_out * H_out * W_out);
    int d = tmp / (H_out * W_out);
    tmp %= (H_out * W_out);
    int h = tmp / W_out;
    int w = tmp % W_out;

    float maxv = -1e38f;
    for (int kd_i = 0; kd_i < kd; ++kd_i) {
        int i_d = d * s0 - p0 + kd_i * d0;
        if (i_d < 0 || i_d >= D_in) continue;
        for (int kh_i = 0; kh_i < kh; ++kh_i) {
            int i_h = h * s1 - p1 + kh_i * d1;
            if (i_h < 0 || i_h >= H_in) continue;
            for (int kw_i = 0; kw_i < kw; ++kw_i) {
                int i_w = w * s2 - p2 + kw_i * d2;
                if (i_w < 0 || i_w >= W_in) continue;
                int i_idx = (((n * C + c) * D_in + i_d) * H_in + i_h) * W_in + i_w;
                float v = input[i_idx];
                if (v > maxv) maxv = v;
            }
        }
    }
    int o_idx = (((n * C + c) * D_out + d) * H_out + h) * W_out + w;
    output[o_idx] = maxv;
}

// ----------------------------------------------------------------------
// 3) Fused reduction: adaptive‑avg‑pool + bias add + sum over channels
// ----------------------------------------------------------------------
__global__ void fused_reduce_kernel(const float* __restrict__ pooled,
                                    const float* __restrict__ bias,
                                    float* __restrict__ output,
                                    const int N, const int C,
                                    const int D_pool, const int H_pool, const int W_pool,
                                    const int Gd, const int Gh, const int Gw) {
    const int total = N * Gd * Gh * Gw;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int tmp = idx;
    int n = tmp / (Gd * Gh * Gw);
    tmp %= (Gd * Gh * Gw);
    int gd = tmp / (Gh * Gw);
    tmp %= (Gh * Gw);
    int gh = tmp / Gw;
    int gw = tmp % Gw;

    // region of the pooled tensor that maps to this global output position
    int s_d = (gd * D_pool) / Gd;
    int e_d = ((gd + 1) * D_pool) / Gd - 1;
    int s_h = (gh * H_pool) / Gh;
    int e_h = ((gh + 1) * H_pool) / Gh - 1;
    int s_w = (gw * W_pool) / Gw;
    int e_w = ((gw + 1) * W_pool) / Gw - 1;

    int region_sz = (e_d - s_d + 1) * (e_h - s_h + 1) * (e_w - s_w + 1);
    float sum_val = 0.0f;
    for (int c = 0; c < C; ++c) {
        float sum_max = 0.0f;
        for (int pd = s_d; pd <= e_d; ++pd) {
            for (int ph = s_h; ph <= e_h; ++ph) {
                for (int pw = s_w; pw <= e_w; ++pw) {
                    int p_idx = (((n * C + c) * D_pool + pd) * H_pool + ph) * W_pool + pw;
                    sum_max += pooled[p_idx];
                }
            }
        }
        float avg = sum_max / (float)region_sz;
        sum_val += avg + bias[c];
    }
    output[idx] = sum_val;
}

// ----------------------------------------------------------------------
// Wrappers called from Python
// ----------------------------------------------------------------------
void conv3d(torch::Tensor input, torch::Tensor weight, torch::Tensor output,
            int N, int C_in, int D_in, int H_in, int W_in,
            int C_out, int Kd, int Kh, int Kw,
            int s0, int s1, int s2,
            int p0, int p1, int p2,
            int d0, int d1, int d2,
            int D_out, int H_out, int W_out,
            float divisor) {
    int threads = 256;
    int blocks = (N * C_out * D_out * H_out * W_out + threads - 1) / threads;
    conv3d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(),
        N, C_in, D_in, H_in, W_in,
        C_out, Kd, Kh, Kw,
        s0, s1, s2,
        p0, p1, p2,
        d0, d1, d2,
        D_out, H_out, W_out,
        divisor);
    cudaDeviceSynchronize();
}

void maxpool3d(torch::Tensor input, torch::Tensor output,
               int N, int C,
               int D_in, int H_in, int W_in,
               int D_out, int H_out, int W_out,
               int kd, int kh, int kw,
               int s0, int s1, int s2,
               int p0, int p1, int p2,
               int d0, int d1, int d2) {
    int threads = 256;
    int blocks = (N * C * D_out * H_out * W_out + threads - 1) / threads;
    maxpool3d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        N, C, D_in, H_in, W_in,
        D_out, H_out, W_out,
        kd, kh, kw,
        s0, s1, s2,
        p0, p1, p2,
        d0, d1, d2);
    cudaDeviceSynchronize();
}

void fused_reduce(torch::Tensor pooled, torch::Tensor bias, torch::Tensor output,
                  int N, int C,
                  int D_pool, int H_pool, int W_pool,
                  int Gd, int Gh, int Gw) {
    int threads = 256;
    int blocks = (N * Gd * Gh * Gw + threads - 1) / threads;
    fused_reduce_kernel<<<blocks, threads>>>(
        pooled.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(),
        N, C, D_pool, H_pool, W_pool,
        Gd, Gh, Gw);
    cudaDeviceSynchronize();
}
"""

# ----------------------------------------------------------------------
# C++ binding – exposes the three wrappers to Python
# ----------------------------------------------------------------------
cpp_src = r"""
#include <torch/extension.h>

void conv3d(torch::Tensor input, torch::Tensor weight, torch::Tensor output,
            int N, int C_in, int D_in, int H_in, int W_in,
            int C_out, int Kd, int Kh, int Kw,
            int s0, int s1, int s2,
            int p0, int p1, int p2,
            int d0, int d1, int d2,
            int D_out, int H_out, int W_out,
            float divisor);

void maxpool3d(torch::Tensor input, torch::Tensor output,
               int N, int C,
               int D_in, int H_in, int W_in,
               int D_out, int H_out, int W_out,
               int kd, int kh, int kw,
               int s0, int s1, int s2,
               int p0, int p1, int p2,
               int d0, int d1, int d2);

void fused_reduce(torch::Tensor pooled, torch::Tensor bias, torch::Tensor output,
                  int N, int C,
                  int D_pool, int H_pool, int W_pool,
                  int Gd, int Gh, int Gw);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv3d",   &conv3d,   "3‑D convolution");
    m.def("maxpool3d",&maxpool3d,"3‑D max‑pooling");
    m.def("fused_reduce", &fused_reduce,
          "Adaptive avg‑pool + bias + sum over channels");
}
"""

# ----------------------------------------------------------------------
# Compile the extension
# ----------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_ops',
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# ----------------------------------------------------------------------
# Helpers for shape computation
# ----------------------------------------------------------------------
def conv_out_dim(in_dim, kernel, stride, padding, dilation):
    return (in_dim + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1

def maxpool_out_dim(in_dim, kernel, stride, padding, dilation, ceil_mode):
    if ceil_mode:
        return (in_dim + 2 * padding - dilation * (kernel - 1) - 1 + stride - 1) // stride + 1
    else:
        return (in_dim + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1

# ----------------------------------------------------------------------
# The functional model that will be evaluated
# ----------------------------------------------------------------------
def functional_model(
    x,
    *,
    conv_weight,
    conv_bias,          # not used – bias is added after pooling
    conv_stride,
    conv_padding,
    conv_dilation,
    conv_groups,        # ignored (only groups==1 is implemented)
    max_pool_kernel_size,
    max_pool_stride,
    max_pool_padding,
    max_pool_dilation,
    max_pool_ceil_mode,
    max_pool_return_indices,  # ignored
    global_avg_pool_output_size,
    divisor,
    bias,
    sum_dim,
):
    # Keep data contiguous for kernel launches
    x = x.contiguous()
    conv_weight = conv_weight.contiguous()
    bias = bias.contiguous()

    # ------------------------------------------------------------------
    # 1) Convolution (plus division by divisor)
    # ------------------------------------------------------------------
    N, C_in, D_in, H_in, W_in = x.shape

    # unpack stride / padding / dilation (allow single int or 3‑tuple)
    stride = conv_stride if isinstance(conv_stride, (tuple, list)) else (conv_stride, conv_stride, conv_stride)
    padding = conv_padding if isinstance(conv_padding, (tuple, list)) else (conv_padding, conv_padding, conv_padding)
    dilation = conv_dilation if isinstance(conv_dilation, (tuple, list)) else (conv_dilation, conv_dilation, conv_dilation)

    C_out, C_in_w, Kd, Kh, Kw = conv_weight.shape   # weight shape: (out, in, kd, kh, kw)

    D_out = conv_out_dim(D_in, Kd, stride[0], padding[0], dilation[0])
    H_out = conv_out_dim(H_in, Kh, stride[1], padding[1], dilation[1])
    W_out = conv_out_dim(W_in, Kw, stride[2], padding[2], dilation[2])

    conv_out = torch.empty(N, C_out, D_out, H_out, W_out, dtype=torch.float32, device='cuda')

    fused_ext.conv3d(
        x, conv_weight, conv_out,
        N, C_in, D_in, H_in, W_in,
        C_out, Kd, Kh, Kw,
        stride[0], stride[1], stride[2],
        padding[0], padding[1], padding[2],
        dilation[0], dilation[1], dilation[2],
        D_out, H_out, W_out,
        float(divisor)
    )

    # ------------------------------------------------------------------
    # 2) Max‑pooling
    # ------------------------------------------------------------------
    pool_k = max_pool_kernel_size if isinstance(max_pool_kernel_size, (tuple, list)) else (max_pool_kernel_size, max_pool_kernel_size, max_pool_kernel_size)
    pool_s = max_pool_stride    if isinstance(max_pool_stride,    (tuple, list)) else (max_pool_stride,    max_pool_stride,    max_pool_stride)
    pool_p = max_pool_padding   if isinstance(max_pool_padding,   (tuple, list)) else (max_pool_padding,   max_pool_padding,   max_pool_padding)
    pool_d = max_pool_dilation  if isinstance(max_pool_dilation,  (tuple, list)) else (max_pool_dilation,  max_pool_dilation,  max_pool_dilation)
    ceil_flag = 1 if max_pool_ceil_mode else 0

    D_pool = maxpool_out_dim(D_out, pool_k[0], pool_s[0], pool_p[0], pool_d[0], max_pool_ceil_mode)
    H_pool = maxpool_out_dim(H_out, pool_k[1], pool_s[1], pool_p[1], pool_d[1], max_pool_ceil_mode)
    W_pool = maxpool_out_dim(W_out, pool_k[2], pool_s[2], pool_p[2], pool_d[2], max_pool_ceil_mode)

    pooled = torch.empty(N, C_out, D_pool, H_pool, W_pool, dtype=torch.float32, device='cuda')

    fused_ext.maxpool3d(
        conv_out, pooled,
        N, C_out,
        D_out, H_out, W_out,
        D_pool, H_pool, W_pool,
        pool_k[0], pool_k[1], pool_k[2],
        pool_s[0], pool_s[1], pool_s[2],
        pool_p[0], pool_p[1], pool_p[2],
        pool_d[0], pool_d[1], pool_d[2]
    )

    # ------------------------------------------------------------------
    # 3) Fused reduction: adaptive‑avg‑pool + bias + sum over channels
    # ------------------------------------------------------------------
    Gd, Gh, Gw = global_avg_pool_output_size if isinstance(global_avg_pool_output_size, (tuple, list)) else (global_avg_pool_output_size, global_avg_pool_output_size, global_avg_pool_output_size)

    out = torch.empty(N, Gd, Gh, Gw, dtype=torch.float32, device='cuda')

    fused_ext.fused_reduce(
        pooled, bias, out,
        N, C_out,
        D_pool, H_pool, W_pool,
        Gd, Gh, Gw
    )

    # The original program sums over dim=1 (the channel dimension).  In our
    # fused kernel we already performed that sum, therefore the result already
    # has shape (batch, Gd, Gh, Gw).  If the original code ever used a different
    # sum_dim we would need an extra kernel – here we simply assert the common case.
    assert sum_dim == 1, "Only sum_dim==1 is implemented in the fused kernel"
    return out

# ----------------------------------------------------------------------
# Dummy helpers for the harness that may call get_init_inputs / get_inputs
# ----------------------------------------------------------------------
def get_init_inputs():
    # Example values that match the original test
    return [
        8,                     # in_channels
        16,                    # out_channels
        (3, 3, 3),             # kernel_size
        2.0,                   # divisor
        (2, 2, 2),             # pool_size
        (16, 1, 1, 1),         # bias_shape (not used directly)
        1                      # sum_dim
    ]

def get_inputs():
    # Return a single input tensor of the shape used in the benchmark
    return [torch.rand(128, 8, 16, 64, 64, device='cuda')]
