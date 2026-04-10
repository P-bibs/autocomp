# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_090325/code_3.py
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

# -------------------------------------------------------------------------
# Inline CUDA source – tiled batch matrix multiplication kernel
# -------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// Tile size – 32×32 yields 1024 threads per block (max) and fits easily
// into the 48 KB shared‑memory/L1 cache of an RTX 2080 Ti.
constexpr int TILE = 32;

__global__ void bmm_tiled_kernel(
    const float* __restrict__ A,          // (batch, M, K)
    const float* __restrict__ B,          // (batch, K, N)
    float*       __restrict__ C,          // (batch, M, N)
    const int batch,
    const int M,
    const int K,
    const int N)
{
    // ---- block / thread indices ---------------------------------------
    const int b        = blockIdx.x;               // batch index
    const int row_tile = blockIdx.y;               // tile row of output
    const int col_tile = blockIdx.z;               // tile column of output

    const int ty = threadIdx.y;    // 0 … TILE-1
    const int tx = threadIdx.x;    // 0 … TILE-1

    // coordinates in the full matrix
    const int row = row_tile * TILE + ty;
    const int col = col_tile * TILE + tx;

    // ---- shared memory for the two tiles -------------------------------
    extern __shared__ float s[];
    float* sA = s;                           // tile of A (TILE × TILE)
    float* sB = s + TILE * TILE;             // tile of B (TILE × TILE)

    float acc = 0.0f;

    // number of K‑tiles to cover the K dimension
    const int num_k_tiles = (K + TILE - 1) / TILE;

    // ---- iterate over K‑tiles -----------------------------------------
    for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
        const int k_start = k_tile * TILE;

        // ---- load tile of A -----------------------------------------
        if (row < M && (k_start + tx) < K) {
            // A[b, row, k_start+tx]
            const int a_idx = ((b * M + row) * K + (k_start + tx));
            sA[ty * TILE + tx] = A[a_idx];
        } else {
            sA[ty * TILE + tx] = 0.0f;
        }

        // ---- load tile of B -----------------------------------------
        if (col < N && (k_start + ty) < K) {
            // B[b, k_start+ty, col]
            const int b_idx = ((b * K + (k_start + ty)) * N + col);
            sB[ty * TILE + tx] = B[b_idx];
        } else {
            sB[ty * TILE + tx] = 0.0f;
        }

        __syncthreads();

        // ---- compute partial dot‑product for this tile ---------------
        #pragma unroll
        for (int kk = 0; kk < TILE; ++kk) {
            acc += sA[ty * TILE + kk] * sB[kk * TILE + tx];
        }

        __syncthreads();
    }

    // ---- write back result --------------------------------------------
    if (row < M && col < N) {
        const int c_idx = ((b * M + row) * N + col);
        C[c_idx] = acc;
    }
}

// Wrapper that launches the kernel from Python
void fused_op_forward(const torch::Tensor& A,
                      const torch::Tensor& B,
                      torch::Tensor& C)
{
    const int batch = A.size(0);
    const int M     = A.size(1);
    const int K     = A.size(2);
    const int N     = B.size(2);

    // Grid: (batch, M/TILE, N/TILE)
    dim3 block(TILE, TILE);                     // 32×32 = 1024 threads
    dim3 grid(batch,
              (M + TILE - 1) / TILE,
              (N + TILE - 1) / TILE);

    const size_t shared_mem = 2 * TILE * TILE * sizeof(float);

    bmm_tiled_kernel<<<grid, block, shared_mem>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        batch, M, K, N);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
}
"""

# -------------------------------------------------------------------------
# C++ binding (pybind11) – no usage of the `function` argument
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>
void fused_op_forward(const torch::Tensor& A,
                      const torch::Tensor& B,
                      torch::Tensor& C);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward,
          "Fused batch matrix multiplication forward (tiled CUDA kernel)");
}
"""

# Compile the extension with aggressive optimisation flags
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# -------------------------------------------------------------------------
# Public API required by the evaluation harness
# -------------------------------------------------------------------------
def get_init_inputs():
    """No special initialisation needed."""
    return []

def get_inputs():
    """Re‑create the original problem sizes (as in the benchmark)."""
    batch = 128
    m = 128 * 4   # 512
    k = 256 * 4   # 1024
    n = 512 * 4   # 2048
    A = torch.rand(batch, m, k, dtype=torch.float32, device='cuda')
    B = torch.rand(batch, k, n, dtype=torch.float32, device='cuda')
    return [A, B]

def functional_model(A, B):
    """Tile‑based batch matrix multiplication – replacement for torch.bmm."""
    # Ensure the inputs reside on the GPU
    if not A.is_cuda:
        A = A.to('cuda')
    if not B.is_cuda:
        B = B.to('cuda')

    batch, m, k = A.shape
    _,        _, n = B.shape

    # Allocate output tensor
    C = torch.empty((batch, m, n), dtype=A.dtype, device=A.device)

    # Invoke the custom tiled kernel
    fused_ext.fused_op(A, B, C)

    return C
