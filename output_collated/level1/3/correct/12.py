# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_091341/code_3.py
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




# ----------------------------------------------------------------------
# Optimised batched matrix multiplication – custom CUDA kernel
# ----------------------------------------------------------------------
import torch
from torch.utils.cpp_extension import load_inline

# ----------------------------------------------------------------------
# CUDA source – tiled batched GEMM kernel
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

constexpr int BLOCK_SIZE = 16;          // 16x16 thread block

// ------------------------------------------------------------------
// Tiled batched matrix multiplication kernel
// ------------------------------------------------------------------
__global__ void bmm_kernel(const float* __restrict__ A,
                           const float* __restrict__ B,
                           float*       __restrict__ C,
                           int M, int N, int K, int batch_size)
{
    // batch index
    int batch = blockIdx.z;

    // tile origin in the output matrix
    int tileRow = blockIdx.y * BLOCK_SIZE;
    int tileCol = blockIdx.x * BLOCK_SIZE;

    // thread's row & column within the tile
    int row = tileRow + threadIdx.y;
    int col = tileCol + threadIdx.x;

    // shared memory for the two tiles
    __shared__ float As[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE * BLOCK_SIZE];

    float sum = 0.0f;

    // number of tiles along the K dimension
    int numTiles = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int t = 0; t < numTiles; ++t)
    {
        // ----- load tile from A -----
        int aCol = t * BLOCK_SIZE + threadIdx.x;
        if (row < M && aCol < K)
            As[threadIdx.y * BLOCK_SIZE + threadIdx.x] =
                A[batch * M * K + row * K + aCol];
        else
            As[threadIdx.y * BLOCK_SIZE + threadIdx.x] = 0.0f;

        // ----- load tile from B -----
        int bRow = t * BLOCK_SIZE + threadIdx.y;
        if (bRow < K && col < N)
            Bs[threadIdx.y * BLOCK_SIZE + threadIdx.x] =
                B[batch * K * N + bRow * N + col];
        else
            Bs[threadIdx.y * BLOCK_SIZE + threadIdx.x] = 0.0f;

        __syncthreads();

        // ----- compute partial dot product for this tile -----
        if (row < M && col < N)
        {
            #pragma unroll
            for (int i = 0; i < BLOCK_SIZE; ++i)
                sum += As[threadIdx.y * BLOCK_SIZE + i] *
                       Bs[i * BLOCK_SIZE + threadIdx.x];
        }

        __syncthreads();
    }

    // ----- write back result -----
    if (row < M && col < N)
        C[batch * M * N + row * N + col] = sum;
}

// ------------------------------------------------------------------
// Host wrapper that launches the kernel
// ------------------------------------------------------------------
void bmm_cuda(const torch::Tensor& A,
              const torch::Tensor& B,
              torch::Tensor& C)
{
    int batch = A.size(0);
    int M     = A.size(1);
    int K     = A.size(2);
    int N     = B.size(2);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid( (N + BLOCK_SIZE - 1) / BLOCK_SIZE,
               (M + BLOCK_SIZE - 1) / BLOCK_SIZE,
               batch );

    bmm_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K, batch);
}
"""

# ----------------------------------------------------------------------
# C++ binding – exposes the kernel to Python
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void bmm_cuda(const torch::Tensor& A,
              const torch::Tensor& B,
              torch::Tensor& C);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bmm_cuda", &bmm_cuda,
          "Batched matrix multiplication on GPU (custom CUDA kernel)");
}
"""

# ----------------------------------------------------------------------
# Compile the inline CUDA extension with aggressive optimisation flags
# ----------------------------------------------------------------------
fused_ext = load_inline(
    name='bmm_cuda',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# ----------------------------------------------------------------------
# Functional model – replaces torch.bmm with the custom kernel
# ----------------------------------------------------------------------
def functional_model(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs C = A @ B using a hand‑tuned CUDA batched‑GEMM kernel.
    The function expects the same interface as the original torch.bmm.
    """
    # Make sure the inputs reside on the GPU
    if not A.is_cuda:
        A = A.cuda()
    if not B.is_cuda:
        B = B.cuda()

    batch, M, K = A.shape
    N = B.shape[2]

    # Allocate output tensor
    C = torch.empty((batch, M, N), dtype=A.dtype, device=A.device)

    # Launch the custom kernel
    fused_ext.bmm_cuda(A, B, C)

    return C


# ----------------------------------------------------------------------
# Helpers required by the harness (not used in evaluation)
# ----------------------------------------------------------------------
def get_init_inputs():
    """No extra initialization inputs are required."""
    return []


def get_inputs():
    """Generate the same random inputs as the original benchmark."""
    batch_size = 128
    m = 128 * 4
    k = 256 * 4
    n = 512 * 4
    A = torch.rand(batch_size, m, k)
    B = torch.rand(batch_size, k, n)
    return [A, B]
