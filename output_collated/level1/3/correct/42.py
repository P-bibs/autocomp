# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_093747/code_3.py
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
# CUDA kernel source – tiled batched matrix‑multiply (FP32)
# -------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define BM 16   // tile height
#define BN 16   // tile width
#define BK 16   // reduction tile size

__global__ void bmm_tiled_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    int batch)
{
    // Shared memory for the two tiles
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    // Map block/thread to output coordinates and batch index
    int batch_idx = blockIdx.z;
    int row       = blockIdx.x * BM + threadIdx.x;   // output row
    int col       = blockIdx.y * BN + threadIdx.y;   // output column

    float sum = 0.0f;

    // Number of K‑tiles to iterate over
    int numKTiles = (K + BK - 1) / BK;

    for (int k_tile = 0; k_tile < numKTiles; ++k_tile) {
        // ----- load tile of A -----
        int a_col = k_tile * BK + threadIdx.y;
        int a_idx = batch_idx * M * K + row * K + a_col;
        if (row < M && a_col < K) {
            As[threadIdx.x][threadIdx.y] = A[a_idx];
        } else {
            As[threadIdx.x][threadIdx.y] = 0.0f;
        }

        // ----- load tile of B -----
        int b_row = k_tile * BK + threadIdx.x;
        int b_idx = batch_idx * K * N + b_row * N + col;
        if (b_row < K && col < N) {
            Bs[threadIdx.x][threadIdx.y] = B[b_idx];
        } else {
            Bs[threadIdx.x][threadIdx.y] = 0.0f;
        }

        __syncthreads();

        // ----- compute partial dot product -----
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            sum += As[threadIdx.x][k] * Bs[k][threadIdx.y];
        }

        __syncthreads();
    }

    // ----- write result -----
    if (row < M && col < N) {
        C[batch_idx * M * N + row * N + col] = sum;
    }
}

// Host wrapper that PyTorch will call
void bmm_tiled(
    const torch::Tensor& A,
    const torch::Tensor& B,
    torch::Tensor& C)
{
    int M = A.size(1);
    int N = B.size(2);
    int K = A.size(2);
    int batch = A.size(0);

    const float* A_data = A.data_ptr<float>();
    const float* B_data = B.data_ptr<float>();
    float*       C_data = C.data_ptr<float>();

    dim3 block(BM, BN);                                   // 256 threads per block
    dim3 grid((M + BM - 1) / BM,
              (N + BN - 1) / BN,
              batch);

    bmm_tiled_kernel<<<grid, block>>>(A_data, B_data, C_data,
                                      M, N, K, batch);
}
"""

# -------------------------------------------------------------------------
# C++ binding (PyBind11) – exposes the function to Python
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void bmm_tiled(
    const torch::Tensor& A,
    const torch::Tensor& B,
    torch::Tensor& C);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bmm_tiled", &bmm_tiled,
          "Tiled batched matrix multiplication (CUDA)");
}
"""

# -------------------------------------------------------------------------
# Compile the inline CUDA extension
# -------------------------------------------------------------------------
bmm_ext = load_inline(
    name='bmm_tiled',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# -------------------------------------------------------------------------
# The only function that will be imported – the optimized functional model
# -------------------------------------------------------------------------
def functional_model(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs batch matrix multiplication C = A @ B using a manually tiled
    CUDA kernel.  Input tensors are expected to be float32 on the GPU.
    """
    # Move inputs to GPU if they are not already there
    if not A.is_cuda:
        A = A.cuda()
    if not B.is_cuda:
        B = B.cuda()

    # Make sure memory layout is contiguous for coalesced accesses
    A = A.contiguous()
    B = B.contiguous()

    batch = A.size(0)
    M = A.size(1)
    K = A.size(2)
    N = B.size(2)

    # Allocate output tensor (same dtype & device as A)
    C = torch.empty((batch, M, N), dtype=A.dtype, device=A.device)

    # Launch the tiled kernel
    bmm_ext.bmm_tiled(A, B, C)

    return C

batch_size = 128
m = 128 * 4
k = 256 * 4
n = 512 * 4

def get_init_inputs():
    return []  # No special initialization inputs needed

def get_inputs():
    A = torch.rand(batch_size, m, k)
    B = torch.rand(batch_size, k, n)
    return [A, B]
