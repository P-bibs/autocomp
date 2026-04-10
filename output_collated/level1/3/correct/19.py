# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_091825/code_3.py
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
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------
# Inline CUDA kernel and PyBind11 binding
# ----------------------------------------------------------------------
from torch.utils.cpp_extension import load_inline

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_M 16
#define TILE_N 16
#define TILE_K 16

__global__ void bmm_kernel(const float* __restrict__ A,
                           const float* __restrict__ B,
                           float* __restrict__ C,
                           int M, int K, int N,
                           int batch)
{
    __shared__ float As[TILE_M][TILE_K];
    __shared__ float Bs[TILE_K][TILE_N];

    int row = blockIdx.x * TILE_M + threadIdx.y;   // output row
    int col = blockIdx.y * TILE_N + threadIdx.x;   // output column
    int b   = blockIdx.z;                           // batch index

    if (row >= M || col >= N) return;

    const float* A_batch = A + b * M * K;
    const float* B_batch = B + b * K * N;
    float*       C_batch = C + b * M * N;

    float sum = 0.0f;

    for (int k = 0; k < K; k += TILE_K) {
        // ----- load tile of A -------------------------------------------------
        int aCol = k + threadIdx.x;
        if (row < M && aCol < K)
            As[threadIdx.y][threadIdx.x] = A_batch[row * K + aCol];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        // ----- load tile of B -------------------------------------------------
        int bRow = k + threadIdx.y;
        if (bRow < K && col < N)
            Bs[threadIdx.y][threadIdx.x] = B_batch[bRow * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // ----- compute partial dot‑product for this tile ----------------------
        #pragma unroll
        for (int i = 0; i < TILE_K; ++i) {
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }

        __syncthreads();
    }

    C_batch[row * N + col] = sum;
}

void bmm_forward(at::Tensor A, at::Tensor B, at::Tensor C)
{
    int M     = static_cast<int>(A.size(1));
    int K     = static_cast<int>(A.size(2));
    int N     = static_cast<int>(B.size(2));
    int batch = static_cast<int>(A.size(0));

    dim3 block(TILE_N, TILE_M);   // 16×16 threads
    dim3 grid((M + TILE_M - 1) / TILE_M,
              (N + TILE_N - 1) / TILE_N,
              batch);

    bmm_kernel<<<grid, block>>>(A.data_ptr<float>(),
                                B.data_ptr<float>(),
                                C.data_ptr<float>(),
                                M, K, N, batch);
}
"""

cpp_source = r"""
#include <torch/extension.h>
void bmm_forward(at::Tensor A, at::Tensor B, at::Tensor C);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bmm_forward", &bmm_forward, "Batched matrix multiply with shared‑memory tiling");
}
"""

fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# ----------------------------------------------------------------------
# The functional model that will be imported
# ----------------------------------------------------------------------
def functional_model(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Batch matrix multiplication C = A @ B.
    Replaces the original torch.bmm with a shared‑memory‑tiled CUDA kernel.
    """
    # Move inputs to GPU if they happen to be on CPU
    if A.device.type == 'cpu':
        A = A.cuda()
        B = B.cuda()

    # Ensure tensors are contiguous (required for raw pointer access)
    A = A.contiguous()
    B = B.contiguous()

    batch, M, K = A.shape
    _, K2, N = B.shape
    # K must match K2 – we trust the caller

    # Allocate output on the same device
    C = torch.empty((batch, M, N), dtype=A.dtype, device=A.device)

    # Launch the custom kernel
    fused_ext.bmm_forward(A, B, C)

    return C


# ----------------------------------------------------------------------
# Helper definitions (kept for completeness, not required by evaluation)
# ----------------------------------------------------------------------
batch_size = 128
m = 128 * 4   # 512
k = 256 * 4   # 1024
n = 512 * 4   # 2048

def get_init_inputs():
    return []

def get_inputs():
    A = torch.rand(batch_size, m, k)
    B = torch.rand(batch_size, k, n)
    return [A, B]
