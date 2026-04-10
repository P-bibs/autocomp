# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_102701/code_3.py
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Optimized CUDA kernel with better memory access patterns
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

__global__ void fused_unsqueeze_multiply_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N,
    int M
) {
    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Total number of elements in output
    int total_elements = N * M;
    
    // Grid-stride loop to handle cases where we have more elements than threads
    for (int i = idx; i < total_elements; i += blockDim.x * gridDim.x) {
        // Calculate row and column indices
        int row = i / M;  // Corresponds to the index in A
        int col = i % M;  // Corresponds to the column in B
        
        // Direct computation without intermediate tensor
        // C[row, col] = A[row] * B[row, col]
        C[i] = A[row] * B[i];
    }
}

__global__ void fused_unsqueeze_multiply_kernel_optimized(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N,
    int M
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Process multiple elements per thread to improve memory throughput
    for (int i = tid; i < N * M; i += stride) {
        int row = i / M;
        C[i] = A[row] * B[i];
    }
}

void fused_unsqueeze_multiply_forward(
    const torch::Tensor& A,
    const torch::Tensor& B,
    torch::Tensor& C,
    int N,
    int M
) {
    // Configure kernel launch parameters
    const int threads_per_block = 256;
    const int blocks_per_grid = std::min(65535, (N * M + threads_per_block - 1) / threads_per_block);
    
    // Launch optimized kernel
    fused_unsqueeze_multiply_kernel_optimized<<<blocks_per_grid, threads_per_block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N,
        M
    );
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("CUDA kernel launch failed: ", cudaGetErrorString(err));
    }
}
"""

# C++ binding code
cpp_source = r"""
#include <torch/extension.h>

void fused_unsqueeze_multiply_forward(
    const torch::Tensor& A,
    const torch::Tensor& B,
    torch::Tensor& C,
    int N,
    int M
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_unsqueeze_multiply", &fused_unsqueeze_multiply_forward, "Fused unsqueeze and multiply operation");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_unsqueeze_multiply',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(A, B):
    N, M = B.shape
    
    # Allocate output tensor
    C = torch.empty(N, M, dtype=torch.float32, device='cuda')
    
    # Ensure inputs are on GPU and correct dtype
    A_gpu = A.to(torch.float32).cuda()
    B_gpu = B.to(torch.float32).cuda()
    
    # Call custom CUDA kernel
    fused_ext.fused_unsqueeze_multiply(A_gpu, B_gpu, C, N, M)
    
    return C

M = 4096
N = 4096

def get_init_inputs():
    return []  # No special initialization inputs needed

def get_inputs():
    A = torch.rand(N)
    B = torch.rand(N, M)
    return [A, B]
