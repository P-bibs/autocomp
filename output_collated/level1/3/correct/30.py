# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_092739/code_2.py
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
# 1. CUDA kernel: tiled batched GEMM (float32)
# ----------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#ifndef TILE_M
#define TILE_M 16          // rows of C tile
#endif
#ifndef TILE_N
#define TILE_N 16          // cols of C tile
#endif
#ifndef TILE_K
#define TILE_K 16          // reduction dimension tile
#endif

// ---------------------------------------------------------------
// Kernel: each block works on one (batch, tile_row, tile_col)
// ---------------------------------------------------------------
__global__ void batched_gemm_kernel(
    const float* __restrict__ A,   // [B, M, K]
    const float* __restrict__ B,   // [B, K, N]
    float* __restrict__ C,         // [B, M, N]
    const int BATCH,
    const int M,
    const int K,
    const int N)
{
    // ------------------------------------------------------------------
    //  block indices
    // ------------------------------------------------------------------
    const int batch_idx = blockIdx.z;               // which matrix in the batch
    const int tile_row  = blockIdx.x;               // which tile (M dimension)
    const int tile_col  = blockIdx.y;               // which tile (N dimension)

    // ------------------------------------------------------------------
    //  thread indices inside the tile
    // ------------------------------------------------------------------
    const int thread_row = threadIdx.x;   // 0 .. TILE_M-1
    const int thread_col = threadIdx.y;   // 0 .. TILE_N-1

    // ------------------------------------------------------------------
    //  Compute the global output coordinates this thread will write
    // ------------------------------------------------------------------
    const int row = tile_row * TILE_M + thread_row;   // 0 .. M-1
    const int col = tile_col * TILE_N + thread_col;   // 0 .. N-1

    // ------------------------------------------------------------------
    //  Pointers to the beginning of the batch element
    // ------------------------------------------------------------------
    const float* A_batch = A + batch_idx * (size_t)M * K;
    const float* B_batch = B + batch_idx * (size_t)K * N;
    float*       C_batch = C + batch_idx * (size_t)M * N;

    // ------------------------------------------------------------------
    //  Accumulator in registers
    // ------------------------------------------------------------------
    float acc = 0.0f;

    // ------------------------------------------------------------------
    //  Shared memory tiles
    // ------------------------------------------------------------------
    __shared__ float As[TILE_M][TILE_K];
    __shared__ float Bs[TILE_K][TILE_N];

    // ------------------------------------------------------------------
    //  Loop over the reduction dimension K in steps of TILE_K
    // ------------------------------------------------------------------
    const int num_tiles_k = (K + TILE_K - 1) / TILE_K;
    for (int t = 0; t < num_tiles_k; ++t) {
        // Load one element of A into shared memory
        int a_row = row;
        int a_col = t * TILE_K + thread_col;   // reuse thread_col as col inside tile
        if (a_row < M && a_col < K) {
            As[thread_row][thread_col] = A_batch[a_row * K + a_col];
        } else {
            As[thread_row][thread_col] = 0.0f;
        }

        // Load one element of B into shared memory
        int b_row = t * TILE_K + thread_row;   // reuse thread_row as row inside tile
        int b_col = col;
        if (b_row < K && b_col < N) {
            Bs[thread_row][thread_col] = B_batch[b_row * N + b_col];
        } else {
            Bs[thread_row][thread_col] = 0.0f;
        }

        __syncthreads();

        // Compute partial product for this tile
        #pragma unroll
        for (int k = 0; k < TILE_K; ++k) {
            acc += As[thread_row][k] * Bs[k][thread_col];
        }

        __syncthreads();   // make sure tile is consumed before loading next one
    }

    // ------------------------------------------------------------------
    //  Write the result back to global memory
    // ------------------------------------------------------------------
    if (row < M && col < N) {
        C_batch[row * N + col] = acc;
    }
}

// ----------------------------------------------------------------------
// 2. C++ wrapper that decides launch configuration and calls the kernel
// ----------------------------------------------------------------------
void batched_gemm_forward(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C)
{
    // sanity checks (sizes are guaranteed by Python wrapper)
    const int BATCH = A.size(0);
    const int M = A.size(1);
    const int K = A.size(2);
    const int N = B.size(2);

    const dim3 block(TILE_M, TILE_N);
    const dim3 grid( (M + TILE_M - 1) / TILE_M,
                    (N + TILE_N - 1) / TILE_N,
                    BATCH );

    // raw pointers
    const float* A_ptr = A.data_ptr<float>();
    const float* B_ptr = B.data_ptr<float>();
    float*       C_ptr = C.data_ptr<float>();

    // launch
    batched_gemm_kernel<<<grid, block>>>(
        A_ptr, B_ptr, C_ptr,
        BATCH, M, K, N);

    // check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in batched_gemm_forward: %s\n", cudaGetErrorString(err));
    }
}
"""

# ----------------------------------------------------------------------
# 3. C++ binding (PYBIND11)
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

// Forward declaration of the kernel wrapper defined in the .cu file
void batched_gemm_forward(torch::Tensor A,
                          torch::Tensor B,
                          torch::Tensor C);

// PYBIND11 module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("batched_gemm_forward", &batched_gemm_forward,
          "Batched GEMM (custom tiled CUDA kernel)");
}
"""

# ----------------------------------------------------------------------
# 4. Compile the inline extension
# ----------------------------------------------------------------------
batched_gemm_ext = load_inline(
    name="batched_gemm_ext",
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cflags=['-O3'],
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True,
    verbose=False,
)

# ----------------------------------------------------------------------
# 5. The user-facing functional model
# ----------------------------------------------------------------------
def functional_model(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Compute a batched matrix multiplication using a custom tiled CUDA kernel.
    Shapes:
        A: [batch, M, K]
        B: [batch, K, N]
    Returns:
        C: [batch, M, N]
    """
    # --------------------------------------------------------------
    # Ensure inputs are on CUDA and are contiguous (required by the kernel)
    # --------------------------------------------------------------
    if not A.is_cuda:
        A = A.cuda()
    if not B.is_cuda:
        B = B.cuda()
    A = A.contiguous()
    B = B.contiguous()

    # --------------------------------------------------------------
    # Allocate output tensor (same dtype/device as inputs)
    # --------------------------------------------------------------
    batch, M, K = A.shape
    _, K2, N = B.shape
    assert K == K2, "Inner dimensions must match"

    C = torch.empty((batch, M, N), device=A.device, dtype=A.dtype)

    # --------------------------------------------------------------
    # Call the compiled kernel
    # --------------------------------------------------------------
    batched_gemm_ext.batched_gemm_forward(A, B, C)

    return C

# ----------------------------------------------------------------------
# 6. Helper functions used by the benchmark harness (unchanged)
# ----------------------------------------------------------------------
batch_size = 128
m = 128 * 4        # 512
k = 256 * 4        # 1024
n = 512 * 4        # 2048

def get_init_inputs():
    # No special init required for this kernel
    return []

def get_inputs():
    A = torch.rand(batch_size, m, k, dtype=torch.float32)
    B = torch.rand(batch_size, k, n, dtype=torch.float32)
    return [A, B]

# ----------------------------------------------------------------------
# 7. Optional simple sanity-check (not part of the required output)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    A, B = get_inputs()
    C_custom = functional_model(A, B)
    C_ref    = torch.bmm(A, B)
    print("max absolute error:", (C_custom - C_ref).abs().max().item())
