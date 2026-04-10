# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_093329/code_0.py
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

# CUDA kernel for optimized batch matrix multiplication using shared memory
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void batch_matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int batch_size,
    int m,
    int k,
    int n
) {
    // Shared memory for tiles
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    // Block and thread indices
    int batch_idx = blockIdx.z;
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    // Boundary checks
    if (batch_idx >= batch_size) return;
    
    // Calculate global indices
    int a_row = row;
    int b_col = col;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < (k + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tiles into shared memory
        int a_col = t * TILE_SIZE + threadIdx.x;
        int b_row = t * TILE_SIZE + threadIdx.y;
        
        // Load A tile
        if (a_row < m && a_col < k) {
            As[threadIdx.y][threadIdx.x] = A[batch_idx * m * k + a_row * k + a_col];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Load B tile
        if (b_row < k && b_col < n) {
            Bs[threadIdx.y][threadIdx.x] = B[batch_idx * k * n + b_row * n + b_col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Synchronize threads
        __syncthreads();
        
        // Compute partial sum
        for (int i = 0; i < TILE_SIZE; ++i) {
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }
        
        // Synchronize threads
        __syncthreads();
    }
    
    // Write result
    if (row < m && col < n) {
        C[batch_idx * m * n + row * n + col] = sum;
    }
}

void batch_matmul_forward(
    const torch::Tensor& A,
    const torch::Tensor& B,
    torch::Tensor& C
) {
    int batch_size = A.size(0);
    int m = A.size(1);
    int k = A.size(2);
    int n = B.size(2);
    
    // Grid and block dimensions
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((n + TILE_SIZE - 1) / TILE_SIZE, 
              (m + TILE_SIZE - 1) / TILE_SIZE, 
              batch_size);
    
    batch_matmul_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        batch_size,
        m,
        k,
        n
    );
}
"""

# C++ interface/bindings
cpp_source = r"""
#include <torch/extension.h>

void batch_matmul_forward(
    const torch::Tensor& A,
    const torch::Tensor& B,
    torch::Tensor& C
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("batch_matmul", &batch_matmul_forward, "Batch Matrix Multiplication with Shared Memory");
}
"""

# Compile the extension
bmm_ext = load_inline(
    name='bmm_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(
    A,
    B,
):
    # Create output tensor
    batch_size, m, k = A.shape
    n = B.shape[2]
    C = torch.empty(batch_size, m, n, dtype=A.dtype, device=A.device)
    
    # Call custom CUDA kernel
    bmm_ext.batch_matmul(A, B, C)
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
