# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_092739/code_6.py
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
# 1. CUDA kernel: Tiled Batched GEMM (float32)
# Using 16x16 tiles for shared memory efficiency.
# ----------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void batched_gemm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int M, const int K, const int N)
{
    // Block handles one (M, N) tile in one batch element
    const int batch_idx = blockIdx.z;
    const int tile_row  = blockIdx.x;
    const int tile_col  = blockIdx.y;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // Offsets for this batch
    const float* A_batch = A + batch_idx * (size_t)M * K;
    const float* B_batch = B + batch_idx * (size_t)K * N;
    float*       C_batch = C + batch_idx * (size_t)M * N;

    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    float acc = 0.0f;

    // Loop over k-dimension tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load A tile
        if ((tile_row * TILE_SIZE + tx) < M && (t * TILE_SIZE + ty) < K)
            As[tx][ty] = A_batch[(tile_row * TILE_SIZE + tx) * K + (t * TILE_SIZE + ty)];
        else
            As[tx][ty] = 0.0f;

        // Load B tile
        if ((t * TILE_SIZE + tx) < K && (tile_col * TILE_SIZE + ty) < N)
            Bs[tx][ty] = B_batch[(t * TILE_SIZE + tx) * N + (tile_col * TILE_SIZE + ty)];
        else
            Bs[tx][ty] = 0.0f;

        __syncthreads();

        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            acc += As[tx][k] * Bs[k][ty];
        }
        __syncthreads();
    }

    // Write output
    if ((tile_row * TILE_SIZE + tx) < M && (tile_col * TILE_SIZE + ty) < N) {
        C_batch[(tile_row * TILE_SIZE + tx) * N + (tile_col * TILE_SIZE + ty)] = acc;
    }
}

void batched_gemm_forward(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    const int batch = A.size(0);
    const int M = A.size(1);
    const int K = A.size(2);
    const int N = B.size(2);

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 grid((M + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE, batch);

    batched_gemm_kernel<<<grid, threads>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N
    );
}
"""

cpp_source = r"""
void batched_gemm_forward(torch::Tensor A, torch::Tensor B, torch::Tensor C);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("batched_gemm_forward", &batched_gemm_forward, "Batched GEMM Kernel");
}
"""

# Compile extension
gemm_ext = load_inline(
    name='batched_gemm_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True,
    verbose=False
)

def functional_model(A, B):
    # Ensure contiguous GPU tensors
    A = A.contiguous().cuda()
    B = B.contiguous().cuda()
    
    batch, M, K = A.shape
    _, _, N = B.shape
    
    C = torch.empty((batch, M, N), device=A.device, dtype=A.dtype)
    
    gemm_ext.batched_gemm_forward(A, B, C)
    return C

batch_size = 128
m = 128 * 4
k = 256 * 4
n = 512 * 4

def get_init_inputs():
    return []

def get_inputs():
    return [torch.rand(batch_size, m, k), torch.rand(batch_size, k, n)]
