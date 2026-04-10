# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_090902/code_5.py
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

# Optimization: Tiled Matrix Multiplication using Shared Memory
# TILE_DIM 32 is often better for occupancy on Voltra/Turing architectures (RTX 2080Ti)
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_DIM 32

__global__ void matmul_tiled_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int K, int N) {
    int batch = blockIdx.z;
    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;

    __shared__ float sA[TILE_DIM][TILE_DIM];
    __shared__ float sB[TILE_DIM][TILE_DIM];

    float sum = 0.0f;
    const float* A_ptr = A + batch * (size_t)M * K;
    const float* B_ptr = B + batch * (size_t)K * N;

    for (int k_tile = 0; k_tile < (K + TILE_DIM - 1) / TILE_DIM; ++k_tile) {
        // Load block of A
        if (row < M && (k_tile * TILE_DIM + threadIdx.x) < K)
            sA[threadIdx.y][threadIdx.x] = A_ptr[row * K + (k_tile * TILE_DIM + threadIdx.x)];
        else
            sA[threadIdx.y][threadIdx.x] = 0.0f;

        // Load block of B
        if (col < N && (k_tile * TILE_DIM + threadIdx.y) < K)
            sB[threadIdx.y][threadIdx.x] = B_ptr[(k_tile * TILE_DIM + threadIdx.y) * N + col];
        else
            sB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_DIM; ++k)
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N)
        C[batch * (size_t)M * N + row * N + col] = sum;
}

torch::Tensor matmul_tiled(torch::Tensor A, torch::Tensor B) {
    int B_size = A.size(0);
    int M = A.size(1);
    int K = A.size(2);
    int N = B.size(2);
    auto C = torch::zeros({B_size, M, N}, A.options());

    dim3 threads(TILE_DIM, TILE_DIM);
    dim3 grid((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM, B_size);

    matmul_tiled_kernel<<<grid, threads>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N);
    
    return C;
}
"""

cpp_source = r"""
#include <torch/extension.h>
torch::Tensor matmul_tiled(torch::Tensor A, torch::Tensor B);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_tiled", &matmul_tiled, "Tiled Matrix Multiplication");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='matmul_tiled_opt',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(A, B):
    # Ensure inputs are contiguous and on CUDA
    return fused_ext.matmul_tiled(A.contiguous().cuda(), B.contiguous().cuda())

# Testing parameters
batch_size = 128
m = 128 * 4
k = 256 * 4
n = 512 * 4

def get_inputs():
    return [torch.rand(batch_size, m, k).cuda(), torch.rand(batch_size, k, n).cuda()]
