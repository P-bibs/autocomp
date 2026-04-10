# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_094501/code_0.py
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

# Custom CUDA kernel for matrix multiplication
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 32
#define TILE_SIZE 2

__global__ void matmul_kernel(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
) {
    // Shared memory for tiles
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
    
    // Thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Block indices
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    // Calculate global indices
    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < (N + BLOCK_SIZE - 1) / BLOCK_SIZE; ++t) {
        // Load tiles into shared memory with boundary checks
        if (row < M && t * BLOCK_SIZE + tx < N) {
            As[ty][tx] = A[row * N + t * BLOCK_SIZE + tx];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        if (t * BLOCK_SIZE + ty < N && col < K) {
            Bs[ty][tx] = B[(t * BLOCK_SIZE + ty) * K + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        // Synchronize threads
        __syncthreads();
        
        // Compute partial sum
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        // Synchronize threads
        __syncthreads();
    }
    
    // Write result
    if (row < M && col < K) {
        C[row * K + col] = sum;
    }
}

void matmul_forward(
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& C
) {
    int M = A.size(0);
    int N = A.size(1);
    int K = B.size(1);
    
    // Define grid and block dimensions
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((K + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // Launch kernel
    matmul_kernel<<<gridDim, blockDim>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M,
        N,
        K
    );
}
"""

# C++ bindings
cpp_source = r"""
#include <torch/extension.h>

void matmul_forward(
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& C
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul", &matmul_forward, "Matrix multiplication CUDA kernel");
}
"""

# Compile the extension
matmul_ext = load_inline(
    name='matmul_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math', '-arch=sm_75'],
    with_cuda=True
)

def functional_model(
    A,
    B,
):
    # Move tensors to GPU if not already there
    if not A.is_cuda:
        A = A.cuda()
    if not B.is_cuda:
        B = B.cuda()
    
    # Create output tensor
    C = torch.empty(A.size(0), B.size(1), device=A.device, dtype=A.dtype)
    
    # Call custom CUDA kernel
    matmul_ext.matmul(A, B, C)
    
    return C

M = 16384 * 2
N = 16 * 2

def get_init_inputs():
    return []  # No special initialization inputs needed

def get_inputs():
    A = torch.rand(M, N, device='cuda')
    B = torch.rand(N, M, device='cuda')
    return [A, B]
