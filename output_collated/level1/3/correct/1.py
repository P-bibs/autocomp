# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_090325/code_1.py
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

# Optimization: Using shared memory to reduce global memory bandwidth usage.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void bmm_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, 
                           int batch, int M, int K, int N) {
    int b = blockIdx.z;
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    float acc = 0.0f;
    const float* A_ptr = A + b * M * K;
    const float* B_ptr = B + b * K * N;

    for (int k = 0; k < (K + TILE_SIZE - 1) / TILE_SIZE; ++k) {
        if (row < M && (k * TILE_SIZE + threadIdx.x) < K)
            sA[threadIdx.y][threadIdx.x] = A_ptr[row * K + (k * TILE_SIZE + threadIdx.x)];
        else
            sA[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && (k * TILE_SIZE + threadIdx.y) < K)
            sB[threadIdx.y][threadIdx.x] = B_ptr[(k * TILE_SIZE + threadIdx.y) * N + col];
        else
            sB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; ++i)
            acc += sA[threadIdx.y][i] * sB[i][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N)
        C[b * M * N + row * N + col] = acc;
}

void bmm_cuda(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    int batch = A.size(0);
    int M = A.size(1);
    int K = A.size(2);
    int N = B.size(2);

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE, batch);

    bmm_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), batch, M, K, N);
}
"""

cpp_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

void bmm_cuda(torch::Tensor A, torch::Tensor B, torch::Tensor C);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bmm_cuda", &bmm_cuda, "Batched Matrix Multiplication (CUDA)");
}
"""

bmm_ext = load_inline(
    name='bmm_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(A, B):
    C = torch.empty(A.size(0), A.size(1), B.size(2), device='cuda')
    bmm_ext.bmm_cuda(A.cuda(), B.cuda(), C)
    return C
