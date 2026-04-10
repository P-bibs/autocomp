# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_100346/code_1.py
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
from torch.utils.cpp_extension import load_inline

# Define the CUDA kernel for matrix multiplication with shared memory tiling
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void matmul_shared_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N
) {
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;

    for (int k = 0; k < (N + TILE_SIZE - 1) / TILE_SIZE; ++k) {
        // Load tile of A into shared memory
        if (row < M && (k * TILE_SIZE + tx) < N) {
            sA[ty][tx] = A[row * N + k * TILE_SIZE + tx];
        } else {
            sA[ty][tx] = 0.0f;
        }

        // Load tile of B into shared memory
        if ((k * TILE_SIZE + ty) < N && col < M) {
            sB[ty][tx] = B[(k * TILE_SIZE + ty) * M + col];
        } else {
            sB[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Compute partial sum
        for (int i = 0; i < TILE_SIZE; ++i) {
            sum += sA[ty][i] * sB[i][tx];
        }

        __syncthreads();
    }

    // Write result to global memory
    if (row < M && col < M) {
        C[row * M + col] = sum;
    }
}

void matmul_cuda(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    int M = A.size(0);
    int N = A.size(1);
    
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((M + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    
    matmul_shared_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N
    );
}
"""

# C++ interface binding
cpp_source = r"""
#include <torch/extension.h>

void matmul_cuda(torch::Tensor A, torch::Tensor B, torch::Tensor C);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_cuda", &matmul_cuda, "Matrix multiplication with shared memory");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='matmul_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(A, B):
    M = A.size(0)
    C = torch.empty((M, M), device=A.device, dtype=A.dtype)
    fused_ext.matmul_cuda(A, B, C)
    return C

def get_inputs():
    M = 16384 * 2
    N = 16 * 2
    A = torch.rand(M, N, device='cuda', dtype=torch.float32)
    B = torch.rand(N, M, device='cuda', dtype=torch.float32)
    return [A, B]
