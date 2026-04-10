# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_091341/code_1.py
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

# Optimization: Tiled GEMM using Shared Memory to maximize memory bandwidth
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void bmm_tiled_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, 
                                 int batch_size, int M, int K, int N) {
    int batch_idx = blockIdx.z;
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;
    const float* A_ptr = A + batch_idx * M * K;
    const float* B_ptr = B + batch_idx * K * N;
    float* C_ptr = C + batch_idx * M * N;

    for (int k_tile = 0; k_tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++k_tile) {
        // Load tile of A into shared memory
        int a_row = row;
        int a_col = k_tile * TILE_SIZE + threadIdx.x;
        if (a_row < M && a_col < K) {
            sA[threadIdx.y][threadIdx.x] = A_ptr[a_row * K + a_col];
        } else {
            sA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load tile of B into shared memory
        int b_row = k_tile * TILE_SIZE + threadIdx.y;
        int b_col = col;
        if (b_row < K && b_col < N) {
            sB[threadIdx.y][threadIdx.x] = B_ptr[b_row * N + b_col];
        } else {
            sB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Compute partial sum
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }

        __syncthreads();
    }

    // Write result to global memory
    if (row < M && col < N) {
        C_ptr[row * N + col] = sum;
    }
}

void bmm_tiled_forward(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    int batch_size = A.size(0);
    int M = A.size(1);
    int K = A.size(2);
    int N = B.size(2);

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE, batch_size);

    bmm_tiled_kernel<<<grid, threads>>>(
        A.data_ptr<float>(), 
        B.data_ptr<float>(), 
        C.data_ptr<float>(),
        batch_size, M, K, N
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void bmm_tiled_forward(torch::Tensor A, torch::Tensor B, torch::Tensor C);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bmm_tiled_forward", &bmm_tiled_forward, "Tiled BMM forward");
}
"""

# Compile the extension
bmm_ext = load_inline(
    name='bmm_tiled',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(A, B):
    C = torch.empty(A.size(0), A.size(1), B.size(2), device=A.device, dtype=A.dtype)
    bmm_ext.bmm_tiled_forward(A, B, C)
    return C

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
