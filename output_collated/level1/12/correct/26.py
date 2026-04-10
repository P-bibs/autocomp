# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_101900/code_23.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = []
FORWARD_ARG_NAMES = ['A', 'B']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = []
REQUIRED_FLAT_STATE_NAMES = []


class ModelNew(nn.Module):
    """
    Simple model that performs a matrix multiplication of a diagonal matrix with another matrix.
    C = diag(A) * B
    """

    def __init__(self):
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
# CUDA kernel with shared-memory caching of the row scale factors
# ----------------------------------------------------------------------
# Optimization: #2 Minimize global memory accesses via #3 Shared memory.
# By loading each row-scale factor A[row] into shared memory once per block 
# (16 threads per block row), we reduce global memory pressure significantly.
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_multiply_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ Out,
    int N, int M)
{
    // A_shared caches 16 rows of the A vector. 
    // blockDim.y is 16, matching the number of rows processed per block.
    __shared__ float A_shared[16];

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Load A[row] into shared memory. 
    // Only threadIdx.x == 0 is needed to load the value for row threadIdx.y.
    if (threadIdx.x == 0 && row < N) {
        A_shared[threadIdx.y] = A[row];
    }
    __syncthreads();

    // Compute element-wise operation using cached shared memory.
    if (row < N && col < M) {
        float a = A_shared[threadIdx.y];
        Out[row * M + col] = a * B[row * M + col];
    }
}

void fused_op_forward(torch::Tensor A, torch::Tensor B, torch::Tensor Out) {
    int N = A.size(0);
    int M = B.size(1);

    // threadsPerBlock (32, 16) provides 512 threads, balancing coalescing and cache reuse.
    dim3 threadsPerBlock(32, 16);
    dim3 numBlocks((M + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    fused_multiply_kernel<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), Out.data_ptr<float>(), N, M);
}
"""

# ----------------------------------------------------------------------
# C++ binding
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(torch::Tensor A, torch::Tensor B, torch::Tensor Out);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused broadcast multiply with shared memory");
}
"""

# Compile the extension inline
fused_ext = load_inline(
    name='fused_mul_optimized',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True,
)

def functional_model(A, B):
    """
    Performs out = A.unsqueeze(1) * B optimization using custom CUDA kernel.
    """
    N = A.shape[0]
    M = B.shape[1]
    
    # Initialize output buffer on the same device
    out = torch.empty((N, M), device=A.device, dtype=A.dtype)
    
    # Call the compiled CUDA extension
    fused_ext.fused_op(A, B, out)
    return out

# --- Requirement placeholders ---
M, N = 4096, 4096

def get_init_inputs():
    return []

def get_inputs():
    A = torch.rand(N, device='cuda')
    B = torch.rand(N, M, device='cuda')
    return [A, B]
