# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_091825/code_7.py
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

# ----------------------------------------------------------------------
# CUDA Kernel and C++ Bindings
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_M 16
#define TILE_N 16
#define TILE_K 16

__global__ void bmm_kernel(const float* __restrict__ A,
                           const float* __restrict__ B,
                           float* __restrict__ C,
                           int M, int K, int N)
{
    __shared__ float As[TILE_M][TILE_K];
    __shared__ float Bs[TILE_K][TILE_N];

    int batch_idx = blockIdx.z;
    int row = blockIdx.x * TILE_M + threadIdx.y;
    int col = blockIdx.y * TILE_N + threadIdx.x;

    const float* A_ptr = A + batch_idx * M * K;
    const float* B_ptr = B + batch_idx * K * N;
    float*       C_ptr = C + batch_idx * M * N;

    float sum = 0.0f;

    for (int k = 0; k < K; k += TILE_K) {
        // Load tile into shared memory
        if (row < M && (k + threadIdx.x) < K)
            As[threadIdx.y][threadIdx.x] = A_ptr[row * K + (k + threadIdx.x)];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        if ((k + threadIdx.y) < K && col < N)
            Bs[threadIdx.y][threadIdx.x] = B_ptr[(k + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Compute dot product for this tile
        #pragma unroll
        for (int i = 0; i < TILE_K; ++i) {
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C_ptr[row * N + col] = sum;
    }
}

void bmm_forward(at::Tensor A, at::Tensor B, at::Tensor C) {
    int batch = A.size(0);
    int M = A.size(1);
    int K = A.size(2);
    int N = B.size(2);

    dim3 block(TILE_N, TILE_M);
    dim3 grid((M + TILE_M - 1) / TILE_M, (N + TILE_N - 1) / TILE_N, batch);

    bmm_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N);
}
"""

cpp_source = r"""
#include <torch/extension.h>
void bmm_forward(at::Tensor A, at::Tensor B, at::Tensor C);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bmm_forward", &bmm_forward, "Batched matrix multiply with shared-memory tiling");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='bmm_optimized',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(A, B):
    # Ensure contiguous for coalesced access pattern
    A = A.contiguous().cuda()
    B = B.contiguous().cuda()
    
    batch, m, k = A.shape
    _, _, n = B.shape
    
    C = torch.empty((batch, m, n), dtype=A.dtype, device=A.device)
    fused_ext.bmm_forward(A, B, C)
    return C

# Constants for test harness
batch_size = 128
m = 128 * 4
k = 256 * 4
n = 512 * 4

def get_init_inputs():
    return []

def get_inputs():
    A = torch.rand(batch_size, m, k)
    B = torch.rand(batch_size, k, n)
    return [A, B]
