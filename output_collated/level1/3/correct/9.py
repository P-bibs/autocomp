# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_090902/code_6.py
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

# ------------------------------------------------------------------
# CUDA kernel – tiled batched GEMM using Shared Memory.
# This implementation performs tiling in the K dimension.
# Each block computes a tile of the result matrix for a given batch.
# ------------------------------------------------------------------
cuda_kernel = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

__global__ void batched_gemm_forward_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int BATCH, int M, int N, int K)
{
    // Shared memory for tiles of A and B
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int batch = blockIdx.z;
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float acc = 0.0f;

    // Pointer arithmetic for batch slices
    const float* A_ptr = A + batch * (size_t)M * K;
    const float* B_ptr = B + batch * (size_t)K * N;
    float* C_ptr = C + batch * (size_t)M * N;

    for (int k_tile = 0; k_tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++k_tile) {
        // Load tile into shared memory
        int k_idx = k_tile * TILE_SIZE + threadIdx.x;
        if (row < M && k_idx < K)
            As[threadIdx.y][threadIdx.x] = A_ptr[row * K + k_idx];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        k_idx = k_tile * TILE_SIZE + threadIdx.y;
        if (k_idx < K && col < N)
            Bs[threadIdx.y][threadIdx.x] = B_ptr[k_idx * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Accumulate
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C_ptr[row * N + col] = acc;
    }
}

void batched_gemm(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    int BATCH = A.size(0);
    int M = A.size(1);
    int K = A.size(2);
    int N = B.size(2);

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE, BATCH);

    batched_gemm_forward_kernel<<<grid, threads>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), BATCH, M, N, K
    );
}
'''

cpp_source = r'''
void batched_gemm(torch::Tensor A, torch::Tensor B, torch::Tensor C);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("batched_gemm", &batched_gemm, "Batched GEMM Kernel");
}
'''

# Compile the JIT extension
fused_ext = load_inline(
    name='batched_gemm_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3'],
    with_cuda=True
)

def functional_model(A, B):
    # Ensure inputs are contiguous
    A = A.contiguous()
    B = B.contiguous()
    batch_size, m, k = A.shape
    _, _, n = B.shape
    C = torch.empty((batch_size, m, n), device=A.device, dtype=A.dtype)
    fused_ext.batched_gemm(A, B, C)
    return C

batch_size = 128
m = 128 * 4
k = 256 * 4
n = 512 * 4

def get_init_inputs():
    return []

def get_inputs():
    A = torch.rand(batch_size, m, k, device='cuda', dtype=torch.float32)
    B = torch.rand(batch_size, k, n, device='cuda', dtype=torch.float32)
    return [A, B]
