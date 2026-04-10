# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_102701/code_27.py
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

# ------------------------------------------------------------------
# CUDA kernel – optimized with shared-memory caching of input vector A
# ------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// Tile dimensions chosen for 32x32 to fit 1024 threads per block
#define TILE_ROWS 32
#define TILE_COLS 32

__global__ void fused_unsqueeze_multiply_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float*       __restrict__ C,
    int N,
    int M)
{
    // Shared memory buffer to cache A for this tile
    __shared__ float sA[TILE_ROWS];

    int row = blockIdx.y * TILE_ROWS + threadIdx.y;
    int col = blockIdx.x * TILE_COLS + threadIdx.x;

    // Load A[row] into shared memory
    // Only the first column of threads in a block loads from global memory
    if (threadIdx.x == 0 && row < N) {
        sA[threadIdx.y] = A[row];
    }
    __syncthreads();

    // Perform fused operation
    if (row < N && col < M) {
        C[row * M + col] = sA[threadIdx.y] * B[row * M + col];
    }
}

void fused_unsqueeze_multiply_forward(
    const torch::Tensor& A,
    const torch::Tensor& B,
    torch::Tensor& C,
    int N,
    int M)
{
    dim3 block(TILE_COLS, TILE_ROWS);
    dim3 grid((M + TILE_COLS - 1) / TILE_COLS, (N + TILE_ROWS - 1) / TILE_ROWS);

    fused_unsqueeze_multiply_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N,
        M
    );

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
    int M);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_unsqueeze_multiply", &fused_unsqueeze_multiply_forward, "Fused unsqueeze and multiply operation");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_unsqueeze_multiply_opt',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(A, B):
    """
    Computes C = A.unsqueeze(1) * B using a fused CUDA kernel with shared memory.
    Ensures optimal memory throughput by staging A in shared memory per tile.
    """
    N, M = B.shape
    
    # Ensure inputs are contiguous float32 tensors on GPU
    A_gpu = A.requires_grad_(False).contiguous().to(device='cuda', dtype=torch.float32)
    B_gpu = B.requires_grad_(False).contiguous().to(device='cuda', dtype=torch.float32)
    
    # Allocate output tensor
    C = torch.empty(N, M, dtype=torch.float32, device='cuda')
    
    # Launch specialized CUDA kernel
    fused_ext.fused_unsqueeze_multiply(A_gpu, B_gpu, C, N, M)
    
    return C

def get_init_inputs():
    return []

def get_inputs():
    N, M = 4096, 4096
    A = torch.rand(N)
    B = torch.rand(N, M)
    return [A, B]
