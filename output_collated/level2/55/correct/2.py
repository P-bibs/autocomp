# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_153614/code_7.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_features', 'out_features', 'kernel_size', 'scale_factor']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['matmul_weight', 'matmul_bias', 'max_pool_kernel_size', 'max_pool_stride', 'max_pool_padding', 'max_pool_dilation', 'max_pool_ceil_mode', 'max_pool_return_indices', 'scale_factor']
REQUIRED_FLAT_STATE_NAMES = ['matmul_weight', 'matmul_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs matrix multiplication, max pooling, sum, and scaling.
    """

    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.max_pool = nn.MaxPool1d(kernel_size)
        self.scale_factor = scale_factor

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
    # State for matmul (nn.Linear)
    if 'matmul_weight' in flat_state:
        state_kwargs['matmul_weight'] = flat_state['matmul_weight']
    else:
        state_kwargs['matmul_weight'] = getattr(model.matmul, 'weight', None)
    if 'matmul_bias' in flat_state:
        state_kwargs['matmul_bias'] = flat_state['matmul_bias']
    else:
        state_kwargs['matmul_bias'] = getattr(model.matmul, 'bias', None)
    # State for max_pool (nn.MaxPool1d)
    state_kwargs['max_pool_kernel_size'] = model.max_pool.kernel_size
    state_kwargs['max_pool_stride'] = model.max_pool.stride
    state_kwargs['max_pool_padding'] = model.max_pool.padding
    state_kwargs['max_pool_dilation'] = model.max_pool.dilation
    state_kwargs['max_pool_ceil_mode'] = model.max_pool.ceil_mode
    state_kwargs['max_pool_return_indices'] = model.max_pool.return_indices
    if 'scale_factor' in flat_state:
        state_kwargs['scale_factor'] = flat_state['scale_factor']
    else:
        state_kwargs['scale_factor'] = getattr(model, 'scale_factor')
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
# CUDA source – two fused kernels and their host wrappers
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

constexpr int BLOCK_SIZE = 256;

/* --------------------------------------------------------------
 * GEMV kernel – computes  y = x @ W^T + b   for a whole batch
 * -------------------------------------------------------------- */
__global__ void gemv_kernel(
    const float* __restrict__ x,       // (batch, in_features)
    const float* __restrict__ weight,  // (out_features, in_features)
    const float* __restrict__ bias,    // (out_features) or nullptr
    float* __restrict__ y,             // (batch, out_features)
    int batch_size,
    int in_features,
    int out_features)
{
    int out_idx = blockIdx.y * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.x;

    if (batch_idx >= batch_size || out_idx >= out_features) return;

    const float* w_row = weight + out_idx * in_features;
    const float* x_batch = x + batch_idx * in_features;

    float sum = 0.0f;
    // Vectorised loads – in_features is guaranteed to be a multiple of 4
    for (int j = 0; j < in_features; j += 4) {
        float4 w = __ldg((const float4*)(w_row + j));
        float4 xv = __ldg((const float4*)(x_batch + j));
        sum += w.x * xv.x + w.y * xv.y + w.z * xv.z + w.w * xv.w;
    }
    if (bias) sum += bias[out_idx];
    y[batch_idx * out_features + out_idx] = sum;
}

/* --------------------------------------------------------------
 * Pool‑and‑sum kernel – max‑pool (kernel_size) over the linear
 * output, then sum and apply the scale factor.
 * -------------------------------------------------------------- */
__global__ void pool_sum_kernel(
    const float* __restrict__ y,    // (batch, out_features)
    float* __restrict__ result,     // (batch)
    float scale,
    int batch_size,
    int out_features,
    int kernel_size)
{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int pair_idx = blockIdx.x * blockDim.x + tid;
    int batch_idx = blockIdx.y;

    if (batch_idx >= batch_size) return;

    int num_pairs = out_features / kernel_size;
    float val = 0.0f;

    if (pair_idx < num_pairs) {
        int i = pair_idx * kernel_size;
        // max over the window
        val = y[batch_idx * out_features + i];
        for (int k = 1; k < kernel_size; ++k) {
            val = fmaxf(val, y[batch_idx * out_features + i + k]);
        }
        val *= scale;
    }
    sdata[tid] = val;
    __syncthreads();

    // block‑wide reduction
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // one atomic add per block
    if (tid == 0) {
        atomicAdd(&result[batch_idx], sdata[0]);
    }
}

/* --------------------------------------------------------------
 * Host wrappers called from Python
 * -------------------------------------------------------------- */
void gemv(const torch::Tensor& x,
          const torch::Tensor& weight,
          const torch::Tensor& bias,
          const torch::Tensor& y)
{
    int B = x.size(0);
    int I = x.size(1);
    int O = weight.size(0);

    const int block_size = BLOCK_SIZE;
    dim3 grid(B, (O + block_size - 1) / block_size);
    gemv_kernel<<<grid, block_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        y.data_ptr<float>(),
        B, I, O);
}

void pool_sum(const torch::Tensor& y,
              const torch::Tensor& result,
              float scale,
              int kernel_size,
              int stride)          // stride is ignored – we assume stride == kernel_size
{
    int B = result.size(0);
    int O = y.size(1);
    const int block_size = BLOCK_SIZE;
    int num_pairs = O / kernel_size;
    int grid_x = (num_pairs + block_size - 1) / block_size;
    dim3 grid(grid_x, B);
    pool_sum_kernel<<<grid, block_size, block_size * sizeof(float)>>>(
        y.data_ptr<float>(),
        result.data_ptr<float>(),
        scale,
        B, O, kernel_size);
}
"""

# ----------------------------------------------------------------------
# C++ binding – exposes the two kernels to Python
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void gemv(const torch::Tensor& x,
          const torch::Tensor& weight,
          const torch::Tensor& bias,
          const torch::Tensor& y);

void pool_sum(const torch::Tensor& y,
              const torch::Tensor& result,
              float scale,
              int kernel_size,
              int stride);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemv",   &gemv,   "Custom GEMV (linear) kernel");
    m.def("pool_sum", &pool_sum, "Fused max‑pool + sum + scale kernel");
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
# The functional model that will be imported / evaluated
# ----------------------------------------------------------------------
def functional_model(
    x,
    *,
    matmul_weight,
    matmul_bias,
    max_pool_kernel_size,
    max_pool_stride,
    max_pool_padding,
    max_pool_dilation,
    max_pool_ceil_mode,
    max_pool_return_indices,
    scale_factor,
):
    # Move everything to the GPU
    device = torch.device('cuda')
    x = x.to(device)
    matmul_weight = matmul_weight.to(device)
    if matmul_bias is not None:
        matmul_bias = matmul_bias.to(device)

    batch_size = x.shape[0]
    in_features = x.shape[1]
    out_features = matmul_weight.shape[0]

    # Intermediate buffer for the linear layer
    y = torch.empty((batch_size, out_features), dtype=torch.float32, device=device)

    # 1) fused linear layer
    fused_ext.gemv(x, matmul_weight, matmul_bias, y)

    # Result buffer – must be zero‑initialised because we use atomicAdd
    result = torch.zeros(batch_size, dtype=torch.float32, device=device)

    # 2) fused max‑pool + sum + scale
    #    stride is ignored – we assume non‑overlapping windows (stride == kernel_size)
    fused_ext.pool_sum(y, result, scale_factor,
                       max_pool_kernel_size, max_pool_stride)

    return result

# ----------------------------------------------------------------------
# Helper functions required by the test harness (optional)
# ----------------------------------------------------------------------
batch_size = 128
in_features = 32768
out_features = 32768
kernel_size = 2
scale_factor = 0.5

def get_init_inputs():
    return [in_features, out_features, kernel_size, scale_factor]

def get_inputs():
    return [torch.rand(batch_size, in_features)]
