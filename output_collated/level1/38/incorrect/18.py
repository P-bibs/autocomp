# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_032736/code_2.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = []
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = []
REQUIRED_FLAT_STATE_NAMES = []


class ModelNew(nn.Module):
    """
    Simple model that performs L1 normalization.
    """

    def __init__(self):
        """
        Initializes the L1 normalization layer.
        """
        super(ModelNew, self).__init__()

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
# CUDA source – fused kernel combining reduction and scaling in one pass
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define WARP_SIZE 32
#define MAX_THREADS_PER_BLOCK 256

// Warp-level parallel reduction using shuffle
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// Kernel: fused reduce + normalize per row
__global__ void fused_normalize_kernel(
    const float *x,
    float *output,
    const int batch_size,
    const int dim)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int threads_per_block = blockDim.x;

    // Shared memory for partial sums
    __shared__ float sdata[MAX_THREADS_PER_BLOCK];

    // Step 1: Compute partial sum of |x[row, :]|
    float sum = 0.0f;
    for (int col = tid; col < dim; col += threads_per_block) {
        sum += fabsf(x[row * dim + col]);
    }

    // Store in shared memory
    sdata[tid] = sum;
    __syncthreads();

    // Step 2: Reduce within block using shared memory
    for (int s = threads_per_block / 2; s > WARP_SIZE; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Final warp-level reduction
    if (tid < WARP_SIZE) {
        sdata[tid] = warp_reduce_sum(sdata[tid]);
    }
    __syncthreads();

    // Broadcast final sum to all threads
    float final_sum = sdata[0];

    // Step 3: Normalize elements using the same sum
    for (int col = tid; col < dim; col += threads_per_block) {
        float val = x[row * dim + col];
        output[row * dim + col] = val * static_cast<float>(dim) / final_sum;
    }
}

// Host function
void fused_normalize_forward(torch::Tensor x, torch::Tensor output) {
    const int batch_size = x.size(0);
    const int dim = x.size(1);

    const int threads = min(MAX_THREADS_PER_BLOCK, dim);
    const int blocks = batch_size;

    fused_normalize_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        dim);

    cudaDeviceSynchronize();
}
"""

# ----------------------------------------------------------------------
# C++ Binding
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void fused_normalize_forward(torch::Tensor x, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_normalize", &fused_normalize_forward, "Fused normalization kernel");
}
"""

# ----------------------------------------------------------------------
# Compile extension
# ----------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_norm',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# ----------------------------------------------------------------------
# Functional model
# ----------------------------------------------------------------------
def functional_model(x: torch.Tensor) -> torch.Tensor:
    if not x.is_cuda:
        x = x.cuda()
    output = torch.empty_like(x)
    fused_ext.fused_normalize(x, output)
    return output

# ----------------------------------------------------------------------
# Helper functions (unchanged)
# ----------------------------------------------------------------------
def get_init_inputs():
    return []

def get_inputs():
    batch_size = 32768
    dim = 65535
    x = torch.rand(batch_size, dim, dtype=torch.float32)
    return [x]
