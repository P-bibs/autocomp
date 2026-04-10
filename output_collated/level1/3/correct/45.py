# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_093747/code_6.py
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

# --------------------------------------------------------------
#  CUDA kernel (tiled batched GEMM)
# --------------------------------------------------------------
# We use 16x16 or 32x32 tiles. 32x32 is chosen for better reuse.
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

__global__ void batched_gemm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int batch_count,
    const int M,
    const int N,
    const int K)
{
    const int batch = blockIdx.z;
    const int tile_row = blockIdx.y;
    const int tile_col = blockIdx.x;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    __shared__ float s_A[TILE_SIZE][TILE_SIZE];
    __shared__ float s_B[TILE_SIZE][TILE_SIZE];

    const int row = tile_row * TILE_SIZE + ty;
    const int col = tile_col * TILE_SIZE + tx;

    float acc = 0.0f;
    const float* A_ptr = A + (batch * M * K);
    const float* B_ptr = B + (batch * K * N);

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tile into shared memory
        if (row < M && (t * TILE_SIZE + tx) < K)
            s_A[ty][tx] = A_ptr[row * K + (t * TILE_SIZE + tx)];
        else
            s_A[ty][tx] = 0.0f;

        if ((t * TILE_SIZE + ty) < K && col < N)
            s_B[ty][tx] = B_ptr[(t * TILE_SIZE + ty) * N + col];
        else
            s_B[ty][tx] = 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            acc += s_A[ty][k] * s_B[k][tx];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[(batch * M * N) + row * N + col] = acc;
    }
}

void batched_gemm_forward(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    const int batch = A.size(0);
    const int M = A.size(1);
    const int K = A.size(2);
    const int N = B.size(2);

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, 
              (M + TILE_SIZE - 1) / TILE_SIZE, 
              batch);

    batched_gemm_kernel<<<grid, block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
        batch, M, N, K
    );
}
"""

cpp_source = r"""
void batched_gemm_forward(torch::Tensor A, torch::Tensor B, torch::Tensor C);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("batched_gemm", &batched_gemm_forward, "Batched GEMM Kernel");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='batched_gemm',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(A, B):
    batch, m, k = A.shape
    _, _, n = B.shape
    C = torch.empty((batch, m, n), dtype=A.dtype, device=A.device)
    fused_ext.batched_gemm(A, B, C)
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
