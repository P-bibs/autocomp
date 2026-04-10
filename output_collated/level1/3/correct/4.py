# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_090902/code_3.py
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

# ------------------------------------------------------------
# Problem size constants (same as the original script)
# ------------------------------------------------------------
batch_size = 128
m = 128 * 4   # 512
k = 256 * 4   # 1024
n = 512 * 4   # 2048

# ------------------------------------------------------------
# Inline CUDA source – tiled batch‑matrix‑multiply kernel
# ------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

__global__ void bmm_kernel(const float* A, const float* B, float* C,
                           int batch, int M, int K, int N) {
    // Shared memory for the current tiles of A and B
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    // Block / thread indices
    int b   = blockIdx.x;                     // batch index
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.z * TILE_SIZE + threadIdx.x;

    if (b >= batch || row >= M || col >= N) return;

    float sum = 0.0f;

    // Iterate over the K dimension in tiles
    for (int kk = 0; kk < K; kk += TILE_SIZE) {
        // ---- load tile of A into shared memory ----
        int aRow = row;
        int aCol = kk + threadIdx.x;
        if (aRow < M && aCol < K) {
            As[threadIdx.y][threadIdx.x] = A[(b * M + aRow) * K + aCol];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // ---- load tile of B into shared memory ----
        int bRow = kk + threadIdx.y;
        int bCol = col;
        if (bRow < K && bCol < N) {
            Bs[threadIdx.y][threadIdx.x] = B[(b * K + bRow) * N + bCol];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // ---- compute partial dot product for this tile ----
        #pragma unroll
        for (int kk_inner = 0; kk_inner < TILE_SIZE; ++kk_inner) {
            sum += As[threadIdx.y][kk_inner] * Bs[kk_inner][threadIdx.x];
        }

        __syncthreads();
    }

    // Write final result
    C[(b * M + row) * N + col] = sum;
}

// Wrapper that launches the kernel with a proper grid / block config
void bmm_cuda(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& C) {
    const int batch = A.size(0);
    const int M     = A.size(1);
    const int K     = A.size(2);
    const int N     = B.size(2);

    dim3 grid(batch,
              (M + TILE_SIZE - 1) / TILE_SIZE,
              (N + TILE_SIZE - 1) / TILE_SIZE);
    dim3 block(TILE_SIZE, TILE_SIZE);

    bmm_kernel<<<grid, block>>>(A.data_ptr<float>(),
                                B.data_ptr<float>(),
                                C.data_ptr<float>(),
                                batch, M, K, N);
}
"""

# ------------------------------------------------------------
# C++ binding (pybind11) – exposes bmm_cuda to Python
# ------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void bmm_cuda(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& C);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bmm_cuda", &bmm_cuda,
          "Batched matrix multiplication using shared‑memory tiling");
}
"""

# ------------------------------------------------------------
# Compile the inline extension
# ------------------------------------------------------------
bmm_ext = load_inline(
    name='bmm_shared',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# ------------------------------------------------------------
# Functional model – replaces the original torch.bmm
# ------------------------------------------------------------
def functional_model(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Computes batch matrix multiplication A @ B using a custom tiled
    CUDA kernel that exploits shared memory.
    """
    # Move inputs to GPU if they are not already there
    if not A.is_cuda:
        A = A.cuda()
    if not B.is_cuda:
        B = B.cuda()

    # Ensure row‑major contiguous layout (PyTorch default)
    A = A.contiguous()
    B = B.contiguous()

    # Allocate output tensor on the same device
    C = torch.empty((A.size(0), A.size(1), B.size(2)),
                    dtype=torch.float32, device='cuda')

    # Launch the shared‑memory tiled kernel
    bmm_ext.bmm_cuda(A, B, C)

    # Guarantee completion before returning (helpful for benchmarking)
    torch.cuda.synchronize()
    return C


# ------------------------------------------------------------
# Helper functions required by the benchmark harness
# ------------------------------------------------------------
def get_init_inputs():
    """No extra initialization inputs are needed."""
    return []


def get_inputs():
    """
    Generate random input tensors on the CPU.
    The functional_model will move them to the GPU.
    """
    A = torch.rand(batch_size, m, k)   # (B, M, K)
    B = torch.rand(batch_size, k, n)   # (B, K, N)
    return [A, B]
