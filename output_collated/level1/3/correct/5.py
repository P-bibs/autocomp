# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_090902/code_2.py
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




# --------------------------------------------------------------
#  batched_gemm.py
# --------------------------------------------------------------
# This file contains the whole program – the original helper
# functions, the custom CUDA kernel that implements a batched
# matrix‑multiply using shared‑memory tiling, the pybind11
# binding, and the re‑implemented `functional_model`.
# --------------------------------------------------------------

import torch
from torch.utils.cpp_extension import load_inline

# ------------------------------------------------------------------
# CUDA kernel – tiled batched GEMM.
# ------------------------------------------------------------------
# Tile size (TPB = threads per block).  32×32 works well on an RTX 2080Ti.
# Each thread block computes one tile of the output matrix C for a
# single batch element.
# ------------------------------------------------------------------
cuda_kernel = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>

#ifndef TILE_M
#define TILE_M 32   // rows in output tile
#endif
#ifndef TILE_N
#define TILE_N 32   // cols in output tile
#endif
#ifndef TILE_K
#define TILE_K 32   // depth of the reduction tile
#endif

// ---------------------------------------------------------------
//  batched_gemm_forward_kernel
// ---------------------------------------------------------------
// A: (B, M, K)  row‑major (contiguous)  float32
// B: (B, K, N)  row‑major                float32
// C: (B, M, N)  row‑major                float32
// ---------------------------------------------------------------
__global__ void batched_gemm_forward_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int BATCH,
    int M,
    int N,
    int K)
{
    // ----- block identifiers ------------------------------------------------
    // blockIdx.z   : batch index
    // blockIdx.y   : tile row index in output (0 .. M/TILE_M)
    // blockIdx.x   : tile col index in output (0 .. N/TILE_N)
    // threadIdx.{x,y}: position inside the tile
    // -------------------------------------------------------------------------

    const int batch = blockIdx.z;
    const int tile_row = blockIdx.y;
    const int tile_col = blockIdx.x;

    const int row = tile_row * TILE_M + threadIdx.y;   // absolute row in C
    const int col = tile_col * TILE_N + threadIdx.x;   // absolute col in C

    // Pointers to the current batch slice
    const float* A_batch = A + batch * (size_t)M * K;
    const float* B_batch = B + batch * (size_t)K * N;
    float*       C_batch = C + batch * (size_t)M * N;

    // Shared‑memory tiles
    __shared__ float As[TILE_M][TILE_K];
    __shared__ float Bs[TILE_K][TILE_N];

    float acc = 0.0f;   // accumulator for the dot product

    // Loop over K dimension in increments of TILE_K
    for (int tk = 0; tk < (K + TILE_K - 1) / TILE_K; ++tk) {
        // ---- load a tile of A ----
        int a_row = row;
        int a_col = tk * TILE_K + threadIdx.x;
        if (a_row < M && a_col < K)
            As[threadIdx.y][threadIdx.x] = A_batch[a_row * K + a_col];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;   // padding for out‑of‑bounds

        // ---- load a tile of B ----
        int b_row = tk * TILE_K + threadIdx.y;
        int b_col = col;
        if (b_row < K && b_col < N)
            Bs[threadIdx.y][threadIdx.x] = B_batch[b_row * N + b_col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();   // make sure the whole tile is loaded

        // ---- compute partial dot product for this tile ----
#pragma unroll
        for (int i = 0; i < TILE_K; ++i) {
            acc += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }

        __syncthreads();   // before loading the next tile
    }

    // ---- write the result back ----
    if (row < M && col < N) {
        C_batch[row * N + col] = acc;
    }
}

// ------------------------------------------------------------------
//  Host helper that launches the kernel.
// ------------------------------------------------------------------
void batched_gemm_forward(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C)
{
    // All tensors are assumed to be contiguous, float32, on CUDA.
    const int BATCH = A.size(0);
    const int M = A.size(1);
    const int K = A.size(2);
    const int N = B.size(2);

    const dim3 blockDim(TILE_N, TILE_M);               // 32×32 = 1024 threads
    const dim3 gridDim(
        (N + TILE_N - 1) / TILE_N,
        (M + TILE_M - 1) / TILE_M,
        BATCH);

    batched_gemm_forward_kernel<<<gridDim, blockDim>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        BATCH, M, N, K);

    // Propagate possible launch errors.
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err));
    }
}
'''

# ------------------------------------------------------------------
# Pybind11 binding (no extra C++ source needed beyond the kernel)
# ------------------------------------------------------------------
cpp_source = r'''
#include <torch/extension.h>

void batched_gemm_forward(torch::Tensor A,
                          torch::Tensor B,
                          torch::Tensor C);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("batched_gemm_forward", &batched_gemm_forward,
          "Batched GEMM (custom CUDA kernel)");
}
'''

# ------------------------------------------------------------------
# Build the extension.
# ------------------------------------------------------------------
# NOTE: we deliberately do **not** pass extra flags such as -O3 or
# --use_fast_math because the chosen optimisation is *only* the use
# of shared memory (optimisation #2‑4).  Adding those flags would count
# as another optimisation from the list, violating the “exactly one”
# rule.
fused_ext = load_inline(
    name='batched_gemm_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cflags=[],
    extra_cuda_cflags=[],
    with_cuda=True,
    verbose=False,
)

# ------------------------------------------------------------------
# Helper to allocate the output tensor (contiguous, same device/dtype)
# ------------------------------------------------------------------
def _allocate_output(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Create an empty tensor C = A @ B with shape (B, M, N)."""
    assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors"
    batch, m, _ = A.shape
    _, _, n = B.shape
    return torch.empty((batch, m, n), dtype=A.dtype, device=A.device)

# ------------------------------------------------------------------
# New implementation of functional_model – uses the custom kernel.
# ------------------------------------------------------------------
def functional_model(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Batched matrix‑multiply using a shared‑memory tiled kernel.

    The signature mirrors the original `torch.bmm` version so that the
    surrounding benchmark harness can call it unchanged.
    """
    # Sanity checks (kept minimal to avoid overhead in the hot path)
    if A.shape[0] != B.shape[0] or A.shape[2] != B.shape[1]:
        raise RuntimeError("Incompatible batch or inner dimensions for bmm")
    if not A.is_contiguous():
        A = A.contiguous()
    if not B.is_contiguous():
        B = B.contiguous()

    C = _allocate_output(A, B)
    fused_ext.batched_gemm_forward(A, B, C)
    return C

# ------------------------------------------------------------------
# Benchmark utilities – unchanged from the original script.
# ------------------------------------------------------------------
batch_size = 128
m = 128 * 4
k = 256 * 4
n = 512 * 4

def get_init_inputs():
    # No special initialization needed for the kernel.
    return []

def get_inputs():
    A = torch.rand(batch_size, m, k, device='cuda', dtype=torch.float32)
    B = torch.rand(batch_size, k, n, device='cuda', dtype=torch.float32)
    return [A, B]

# ------------------------------------------------------------------
# If run as a script, perform a quick sanity check.
# ------------------------------------------------------------------
if __name__ == "__main__":
    A, B = get_inputs()
    C = functional_model(A, B)
    # Verify against PyTorch's own bmm (within a small tolerance)
    ref = torch.bmm(A, B)
    max_err = (C - ref).abs().max()
    print(f"Max absolute error vs torch.bmm: {max_err.item():.6e}")
    print("Shape:", C.shape)
