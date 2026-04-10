# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_090902/code_0.py
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

# CUDA kernel with tiling optimization
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void batch_matmul_tiled_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int batch_size,
    int m,
    int n,
    int k
) {
    // Tile indices
    int batch_idx = blockIdx.z;
    int tile_row = blockIdx.y * TILE_SIZE;
    int tile_col = blockIdx.x * TILE_SIZE;
    
    // Thread indices within tile
    int thread_row = threadIdx.y;
    int thread_col = threadIdx.x;
    
    // Boundary checks
    if (batch_idx >= batch_size) return;
    
    // Shared memory for tiles
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    // Position in output matrix
    int row = tile_row + thread_row;
    int col = tile_col + thread_col;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < (k + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tile of A into shared memory
        int a_row = row;
        int a_col = t * TILE_SIZE + thread_col;
        if (a_row < m && a_col < k) {
            As[thread_row][thread_col] = A[batch_idx * m * k + a_row * k + a_col];
        } else {
            As[thread_row][thread_col] = 0.0f;
        }
        
        // Load tile of B into shared memory
        int b_row = t * TILE_SIZE + thread_row;
        int b_col = col;
        if (b_row < k && b_col < n) {
            Bs[thread_row][thread_col] = B[batch_idx * k * n + b_row * n + b_col];
        } else {
            Bs[thread_row][thread_col] = 0.0f;
        }
        
        // Synchronize threads
        __syncthreads();
        
        // Compute partial sum
        #pragma unroll
        for (int i = 0; i < TILE_SIZE; ++i) {
            sum += As[thread_row][i] * Bs[i][thread_col];
        }
        
        // Synchronize threads
        __syncthreads();
    }
    
    // Write result
    if (row < m && col < n) {
        C[batch_idx * m * n + row * n + col] = sum;
    }
}

void batch_matmul_tiled_forward(
    const float* A,
    const float* B,
    float* C,
    int batch_size,
    int m,
    int n,
    int k
) {
    dim3 grid_dim(
        (n + TILE_SIZE - 1) / TILE_SIZE,
        (m + TILE_SIZE - 1) / TILE_SIZE,
        batch_size
    );
    dim3 block_dim(TILE_SIZE, TILE_SIZE);
    
    batch_matmul_tiled_kernel<<<grid_dim, block_dim>>>(
        A, B, C, batch_size, m, n, k
    );
}
"""

# C++ interface
cpp_source = r"""
#include <torch/extension.h>

void batch_matmul_tiled_forward(
    const float* A,
    const float* B,
    float* C,
    int batch_size,
    int m,
    int n,
    int k
);

torch::Tensor batch_matmul_tiled(torch::Tensor A, torch::Tensor B) {
    auto batch_size = A.size(0);
    auto m = A.size(1);
    auto k = A.size(2);
    auto n = B.size(2);
    
    auto C = torch::zeros({batch_size, m, n}, A.options());
    
    batch_matmul_tiled_forward(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        batch_size,
        m,
        n,
        k
    );
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("batch_matmul_tiled", &batch_matmul_tiled, "Tiled batch matrix multiplication");
}
"""

# Compile extension
tiled_ext = load_inline(
    name='tiled_bmm',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(A, B):
    return tiled_ext.batch_matmul_tiled(A, B)

batch_size = 128
m = 128 * 4
k = 256 * 4
n = 512 * 4

def get_init_inputs():
    return []

def get_inputs():
    A = torch.rand(batch_size, m, k).cuda()
    B = torch.rand(batch_size, k, n).cuda()
    return [A, B]

