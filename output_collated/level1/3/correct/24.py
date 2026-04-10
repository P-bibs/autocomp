# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_092316/code_0.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = []
FORWARD_ARG_NAMES = ['A', 'B']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = []
REQUIRED_FLAT_STATE_NAMES = []


class ModelNew(nn.Module):
    """
    Performs batched matrix multiplication (C = A * B) where A, B, and C have the same batch dimension.
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# CUDA kernel for optimized batch matrix multiplication using shared memory tiling
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_DIM 32

__global__ void batch_matmul_kernel(
    const float* A,
    const float* B,
    float* C,
    int batch_size,
    int m,
    int n,
    int k
) {
    // Shared memory for tiles of A and B
    __shared__ float tile_A[TILE_DIM][TILE_DIM];
    __shared__ float tile_B[TILE_DIM][TILE_DIM];
    
    int batch_idx = blockIdx.z;
    if (batch_idx >= batch_size) return;
    
    // Calculate global row and column indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < (k + TILE_DIM - 1) / TILE_DIM; ++t) {
        // Load tiles into shared memory
        int idx_A = batch_idx * m * k + row * k + t * TILE_DIM + threadIdx.x;
        int idx_B = batch_idx * k * n + (t * TILE_DIM + threadIdx.y) * n + col;
        
        // Load elements into shared memory with boundary checks
        if (row < m && (t * TILE_DIM + threadIdx.x) < k) {
            tile_A[threadIdx.y][threadIdx.x] = A[idx_A];
        } else {
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        if ((t * TILE_DIM + threadIdx.y) < k && col < n) {
            tile_B[threadIdx.y][threadIdx.x] = B[idx_B];
        } else {
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Synchronize to ensure all threads have loaded their data
        __syncthreads();
        
        // Compute partial sum for this tile
        for (int i = 0; i < TILE_DIM; ++i) {
            sum += tile_A[threadIdx.y][i] * tile_B[i][threadIdx.x];
        }
        
        // Synchronize before loading next tile
        __syncthreads();
    }
    
    // Write result to global memory
    if (row < m && col < n) {
        C[batch_idx * m * n + row * n + col] = sum;
    }
}

void batch_matmul_forward(
    const torch::Tensor& A,
    const torch::Tensor& B,
    torch::Tensor& C,
    int batch_size,
    int m,
    int n,
    int k
) {
    // Define block and grid dimensions
    dim3 block_dim(TILE_DIM, TILE_DIM);
    dim3 grid_dim((n + TILE_DIM - 1) / TILE_DIM, (m + TILE_DIM - 1) / TILE_DIM, batch_size);
    
    // Launch kernel
    batch_matmul_kernel<<<grid_dim, block_dim>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        batch_size,
        m,
        n,
        k
    );
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR(cudaGetErrorString(err));
    }
}
"""

# C++ interface for the CUDA kernel
cpp_source = r"""
#include <torch/extension.h>

void batch_matmul_forward(
    const torch::Tensor& A,
    const torch::Tensor& B,
    torch::Tensor& C,
    int batch_size,
    int m,
    int n,
    int k
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("batch_matmul", &batch_matmul_forward, "Batch Matrix Multiplication CUDA Kernel");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='batch_matmul_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(A, B):
    batch_size = A.size(0)
    m = A.size(1)
    k = A.size(2)
    n = B.size(2)
    
    # Create output tensor
    C = torch.empty(batch_size, m, n, dtype=A.dtype, device=A.device)
    
    # Call the custom CUDA kernel
    fused_ext.batch_matmul(A, B, C, batch_size, m, n, k)
    
    return C

batch_size = 128
m = 128 * 4
k = 256 * 4
n = 512 * 4

def get_init_inputs():
    return []  # No special initialization inputs needed

def get_inputs():
    A = torch.rand(batch_size, m, k, device='cuda')
    B = torch.rand(batch_size, k, n, device='cuda')
    return [A, B]
