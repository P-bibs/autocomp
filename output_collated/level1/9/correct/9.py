# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_095237/code_0.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = []
FORWARD_ARG_NAMES = ['A', 'B']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = []
REQUIRED_FLAT_STATE_NAMES = []


class ModelNew(nn.Module):
    """
    Simple model that performs a single matrix multiplication (C = A * B) where one of the matrices is tall and skinny (M >> N or N >> M)
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

# Define CUDA kernel with shared memory tiling optimization
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define TILE_SIZE 16

__global__ void matmul_tiling_kernel(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
) {
    // Shared memory for tiles
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    // Thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Block indices
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    // Calculate global indices
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tiles into shared memory
        int k = t * TILE_SIZE + tx;
        if (row < M && k < K) {
            As[ty][tx] = A[row * K + k];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        k = t * TILE_SIZE + ty;
        if (col < N && k < K) {
            Bs[ty][tx] = B[k * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        // Synchronize threads
        __syncthreads();
        
        // Compute partial sum
        for (int i = 0; i < TILE_SIZE; ++i) {
            sum += As[ty][i] * Bs[i][tx];
        }
        
        // Synchronize threads
        __syncthreads();
    }
    
    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

void matmul_tiling_forward(
    const torch::Tensor& A,
    const torch::Tensor& B,
    torch::Tensor& C
) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    
    matmul_tiling_kernel<<<gridDim, blockDim>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M,
        N,
        K
    );
}
"""

# Define C++ binding
cpp_source = r"""
#include <torch/extension.h>

void matmul_tiling_forward(
    const torch::Tensor& A,
    const torch::Tensor& B,
    torch::Tensor& C
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_tiling", &matmul_tiling_forward, "Matrix multiplication with tiling optimization");
}
"""

# Compile the extension with optimization flags
fused_ext = load_inline(
    name='matmul_tiling_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(A, B):
    # Create output tensor
    C = torch.empty(A.size(0), B.size(1), dtype=torch.float32, device=A.device)
    # Call optimized CUDA kernel
    fused_ext.matmul_tiling(A, B, C)
    return C

M = 16384 * 2
N = 16 * 2

def get_init_inputs():
    return []  # No special initialization inputs needed

def get_inputs():
    A = torch.rand(M, N, device='cuda')
    B = torch.rand(N, M, device='cuda')
    return [A, B]
