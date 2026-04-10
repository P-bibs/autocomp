# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_095613/code_0.py
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

# CUDA kernel for matrix multiplication
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define TILE_SIZE 32

__global__ void matmul_kernel(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
) {
    // Calculate global thread indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Shared memory for tiles
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tiles into shared memory
        int a_row = row;
        int a_col = t * TILE_SIZE + threadIdx.x;
        if (a_row < M && a_col < K) {
            As[threadIdx.y][threadIdx.x] = A[a_row * K + a_col];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        int b_row = t * TILE_SIZE + threadIdx.y;
        int b_col = col;
        if (b_row < K && b_col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[b_row * N + b_col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial sum
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

void matmul_forward(int blocks_x, int blocks_y, int threads_x, int threads_y, 
                    torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    dim3 blocks(blocks_x, blocks_y);
    dim3 threads(threads_x, threads_y);
    
    matmul_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        A.size(0),
        B.size(1),
        A.size(1)
    );
}
"""

# C++ interface
cpp_source = r"""
#include <torch/extension.h>

void matmul_forward(int blocks_x, int blocks_y, int threads_x, int threads_y, 
                    torch::Tensor A, torch::Tensor B, torch::Tensor C);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul", &matmul_forward, "Matrix multiplication CUDA kernel");
}
"""

# Compile the extension
matmul_ext = load_inline(
    name='matmul_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(A, B):
    # Move tensors to GPU if not already there
    if not A.is_cuda:
        A = A.cuda()
    if not B.is_cuda:
        B = B.cuda()
        
    # Create output tensor
    C = torch.empty(A.size(0), B.size(1), device='cuda', dtype=torch.float32)
    
    # Define kernel launch parameters
    # For M=32768, N=32768 with TILE_SIZE=32
    threads_x = 32
    threads_y = 32
    blocks_x = (B.size(1) + threads_x - 1) // threads_x
    blocks_y = (A.size(0) + threads_y - 1) // threads_y
    
    # Launch kernel
    matmul_ext.matmul(blocks_x, blocks_y, threads_x, threads_y, A, B, C)
    
    return C

M = 16384 * 2
N = 16 * 2

def get_init_inputs():
    return []  # No special initialization inputs needed

def get_inputs():
    A = torch.rand(M, N, device='cuda')
    B = torch.rand(N, M, device='cuda')
    return [A, B]
