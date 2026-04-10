# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_154447/code_3.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['dim']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['dim']
REQUIRED_FLAT_STATE_NAMES = []


class ModelNew(nn.Module):
    """
    Simple model that performs Max reduction over a specific dimension.
    """

    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): The dimension to reduce over.
        """
        super(ModelNew, self).__init__()
        self.dim = dim

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
    if 'dim' in flat_state:
        state_kwargs['dim'] = flat_state['dim']
    else:
        state_kwargs['dim'] = getattr(model, 'dim')
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

# ------------------------------------------------------------
# 1. CUDA source – three kernels (one for each possible dim)
# ------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>

// ------------------------------------------------------------------
// Kernel for reduction over the last dimension (dim == 2)
// Input shape: (B, N, M)  ->  Output shape: (B, N)
// ------------------------------------------------------------------
__global__ void max_dim2_kernel(const float* __restrict__ input,
                                float* __restrict__ output,
                                const int B, const int N, const int M)
{
    int out_idx = blockIdx.x;                     // flat index of (batch, i)
    int batch   = out_idx / N;
    int i       = out_idx % N;
    const float* row = input + (batch * N + i) * M;

    // each thread loads a subset of the M elements
    float local_max = -FLT_MAX;
    for (int j = threadIdx.x; j < M; j += blockDim.x) {
        float v = row[j];
        if (v > local_max) local_max = v;
    }

    // store to shared memory and reduce
    extern __shared__ float sdata[];
    sdata[threadIdx.x] = local_max;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (sdata[threadIdx.x + s] > sdata[threadIdx.x])
                sdata[threadIdx.x] = sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) output[out_idx] = sdata[0];
}

// ------------------------------------------------------------------
// Kernel for reduction over the middle dimension (dim == 1)
// Input shape: (B, N, M)  ->  Output shape: (B, M)
// ------------------------------------------------------------------
__global__ void max_dim1_kernel(const float* __restrict__ input,
                                float* __restrict__ output,
                                const int B, const int N, const int M)
{
    int out_idx = blockIdx.x;                 // flat index of (batch, m)
    int batch   = out_idx / M;
    int m       = out_idx % M;

    float local_max = -FLT_MAX;
    for (int n = threadIdx.x; n < N; n += blockDim.x) {
        float v = input[(batch * N + n) * M + m];
        if (v > local_max) local_max = v;
    }

    extern __shared__ float sdata[];
    sdata[threadIdx.x] = local_max;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (sdata[threadIdx.x + s] > sdata[threadIdx.x])
                sdata[threadIdx.x] = sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) output[out_idx] = sdata[0];
}

// ------------------------------------------------------------------
// Kernel for reduction over the first dimension (dim == 0)
// Input shape: (B, N, M)  ->  Output shape: (N, M)
// ------------------------------------------------------------------
__global__ void max_dim0_kernel(const float* __restrict__ input,
                                float* __restrict__ output,
                                const int B, const int N, const int M)
{
    int out_idx = blockIdx.x;                 // flat index of (i, j)
    int i = out_idx / M;
    int j = out_idx % M;

    float local_max = -FLT_MAX;
    for (int b = threadIdx.x; b < B; b += blockDim.x) {
        float v = input[(b * N + i) * M + j];
        if (v > local_max) local_max = v;
    }

    extern __shared__ float sdata[];
    sdata[threadIdx.x] = local_max;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (sdata[threadIdx.x + s] > sdata[threadIdx.x])
                sdata[threadIdx.x] = sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) output[out_idx] = sdata[0];
}

// ------------------------------------------------------------------
// Host wrapper that dispatches to the appropriate kernel
// ------------------------------------------------------------------
void max_dim_cuda(torch::Tensor input, int dim, torch::Tensor output) {
    const int B = input.size(0);
    const int N = input.size(1);
    const int M = input.size(2);
    const int threads = 256;                     // arbitrary power‑of‑2 warp‑multiple
    if (dim == 2) {
        const int out_size = B * N;
        max_dim2_kernel<<<out_size, threads, threads * sizeof(float)>>>(
            input.data_ptr<float>(), output.data_ptr<float>(), B, N, M);
    } else if (dim == 1) {
        const int out_size = B * M;
        max_dim1_kernel<<<out_size, threads, threads * sizeof(float)>>>(
            input.data_ptr<float>(), output.data_ptr<float>(), B, N, M);
    } else { // dim == 0
        const int out_size = N * M;
        max_dim0_kernel<<<out_size, threads, threads * sizeof(float)>>>(
            input.data_ptr<float>(), output.data_ptr<float>(), B, N, M);
    }
    cudaDeviceSynchronize();
}
"""

# ------------------------------------------------------------
# 2. C++ interface (PYBIND11)
# ------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void max_dim_cuda(torch::Tensor input, int dim, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("max_dim_cuda", &max_dim_cuda,
          "Custom CUDA max reduction along a given dimension");
}
"""

# ------------------------------------------------------------
# 3. Build the inline extension
# ------------------------------------------------------------
max_ext = load_inline(
    name="max_cuda_ext",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    with_cuda=True,
)

# ------------------------------------------------------------
# 4. Helper functions required by the harness (optional)
# ------------------------------------------------------------
def get_init_inputs():
    """Return a dummy list – the actual dimension is passed at runtime."""
    return []

def get_inputs():
    batch_size = 128
    dim1 = 4096
    dim2 = 4095
    # Create the same tensor shape as the original benchmark
    x = torch.rand(batch_size, dim1, dim2)
    return [x]

# ------------------------------------------------------------
# 5. The function that will be imported & evaluated
# ------------------------------------------------------------
def functional_model(x, *, dim):
    """
    Returns the maximum of `x` along `dim`.
    The computation is performed by a hand‑tuned CUDA kernel
    instead of the generic torch.max implementation.
    """
    # Make sure the input lives on the GPU and is contiguous
    if not x.is_cuda:
        x = x.cuda()
    x = x.contiguous()

    B, N, M = x.shape

    # Compute output shape according to the reduction dimension
    if dim == 0:
        out_shape = (N, M)
    elif dim == 1:
        out_shape = (B, M)
    else:               # dim == 2
        out_shape = (B, N)

    # Allocate output tensor on the same device
    output = torch.empty(out_shape, dtype=x.dtype, device=x.device)

    # Dispatch to the custom CUDA kernel
    max_ext.max_dim_cuda(x, dim, output)

    return output
