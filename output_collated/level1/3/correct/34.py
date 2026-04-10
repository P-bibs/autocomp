# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_092739/code_7.py
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
from torch.utils.cpp_extension import load_inline

# -------------------------------------------------------------------------
# CUDA source – Tiled Batched GEMM Kernel
# -------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// Tile dimensions chosen for performance on RTX 2080 Ti (16x16 is memory-efficient)
constexpr int BLOCK_SIZE = 16;

__global__ void bmm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    int batch = blockIdx.z;
    int row = blockIdx.x * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.y * BLOCK_SIZE + threadIdx.x;

    __shared__ float sA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sB[BLOCK_SIZE][BLOCK_SIZE];

    float acc = 0.0f;

    // Loop over the K dimension
    for (int k_idx = 0; k_idx < K; k_idx += BLOCK_SIZE) {
        // Load tile into shared memory with bounds checking
        if (row < M && (k_idx + threadIdx.x) < K) {
            sA[threadIdx.y][threadIdx.x] = A[batch * M * K + row * K + (k_idx + threadIdx.x)];
        } else {
            sA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < N && (k_idx + threadIdx.y) < K) {
            sB[threadIdx.y][threadIdx.x] = B[batch * K * N + (k_idx + threadIdx.y) * N + col];
        } else {
            sB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Compute local tiles
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            acc += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[batch * M * N + row * N + col] = acc;
    }
}

void fused_op_forward(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    int batch = A.size(0);
    int M = A.size(1);
    int K = A.size(2);
    int N = B.size(2);

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((M + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE, batch);

    bmm_kernel<<<grid, threads>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K
    );
}
"""

# -------------------------------------------------------------------------
# C++ Bindings
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(torch::Tensor A, torch::Tensor B, torch::Tensor C);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Tiled Batched GEMM");
}
"""

# Compile extension
fused_ext = load_inline(
    name='fused_bmm',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Optimized batched matrix multiplication using custom CUDA kernel.
    """
    # Ensure inputs are contiguous on GPU
    A = A.to('cuda', memory_format=torch.contiguous_format)
    B = B.to('cuda', memory_format=torch.contiguous_format)
    
    batch, M, K = A.shape
    _, _, N = B.shape
    
    C = torch.empty((batch, M, N), device='cuda', dtype=torch.float32)
    
    fused_ext.fused_op(A, B, C)
    
    return C
