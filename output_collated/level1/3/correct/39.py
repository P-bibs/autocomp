# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_093329/code_5.py
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

# Tiled CUDA kernel implementation
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void matmul_shared_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, 
                                     int M, int K, int N) {
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
        int k_idx = k_tile * TILE_SIZE + threadIdx.x;
        if (row < M && k_idx < K)
            sA[threadIdx.y][threadIdx.x] = A_ptr[row * K + k_idx];
        else
            sA[threadIdx.y][threadIdx.x] = 0.0f;

        int row_b = k_tile * TILE_SIZE + threadIdx.y;
        if (col < N && row_b < K)
            sB[threadIdx.y][threadIdx.x] = B_ptr[row_b * N + col];
        else
            sB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k)
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N)
        C_ptr[row * N + col] = sum;
}

void matmul_cuda(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    int batch = A.size(0);
    int M = A.size(1);
    int K = A.size(2);
    int N = B.size(2);

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE, batch);

    matmul_shared_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N);
}
"""

cpp_source = r"""
void matmul_cuda(torch::Tensor A, torch::Tensor B, torch::Tensor C);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_cuda", &matmul_cuda, "Tiled Matrix Multiplication");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='matmul_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(A, B):
    batch, m, k = A.shape
    n = B.shape[2]
    # Ensure contiguous tensors for efficient global memory access
    C = torch.empty((batch, m, n), device=A.device, dtype=A.dtype)
    fused_ext.matmul_cuda(A.contiguous(), B.contiguous(), C)
    return C

batch_size = 128
m = 128 * 4
k = 256 * 4
n = 512 * 4

def get_init_inputs():
    return []

def get_inputs():
    A = torch.rand(batch_size, m, k, device='cuda')
    B = torch.rand(batch_size, k, n, device='cuda')
    return [A, B]
