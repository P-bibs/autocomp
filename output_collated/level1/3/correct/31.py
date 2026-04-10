# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_092739/code_3.py
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
# CUDA source – tiled batched GEMM kernel + host wrapper
# -------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

constexpr int BLOCK_M = 16;
constexpr int BLOCK_N = 16;
constexpr int BLOCK_K = 16;

__global__ void bmm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    // batch index
    int batch = blockIdx.z;

    // output element coordinates that this thread computes
    int row = blockIdx.x * BLOCK_M + threadIdx.x;   // 0 … M-1
    int col = blockIdx.y * BLOCK_N + threadIdx.y;   // 0 … N-1

    // shared memory tiles
    __shared__ float sA[BLOCK_M][BLOCK_K];
    __shared__ float sB[BLOCK_K][BLOCK_N];

    float acc = 0.0f;

    // loop over K in blocks of BLOCK_K
    for (int k_tile = 0; k_tile < K; k_tile += BLOCK_K) {
        // ---- load tile from A -----------------------------------------
        if (row < M && (k_tile + threadIdx.y) < K) {
            // A[batch, row, k_tile + threadIdx.y]
            sA[threadIdx.x][threadIdx.y] = A[batch * (M * K) + row * K + (k_tile + threadIdx.y)];
        } else {
            sA[threadIdx.x][threadIdx.y] = 0.0f;
        }

        // ---- load tile from B -----------------------------------------
        if (col < N && (k_tile + threadIdx.x) < K) {
            // B[batch, k_tile + threadIdx.x, col]
            sB[threadIdx.x][threadIdx.y] = B[batch * (K * N) + (k_tile + threadIdx.x) * N + col];
        } else {
            sB[threadIdx.x][threadIdx.y] = 0.0f;
        }

        __syncthreads();

        // ---- compute partial dot product -------------------------------
        #pragma unroll
        for (int k = 0; k < BLOCK_K; ++k) {
            acc += sA[threadIdx.x][k] * sB[k][threadIdx.y];
        }

        __syncthreads();
    }

    // ---- store result -------------------------------------------------
    if (row < M && col < N) {
        C[batch * (M * N) + row * N + col] = acc;
    }
}

// Host wrapper that launches the kernel
void fused_op(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    int batch = A.size(0);
    int M     = A.size(1);
    int K     = A.size(2);
    int N     = B.size(2);

    const float* a_ptr = A.data_ptr<float>();
    const float* b_ptr = B.data_ptr<float>();
    float*       c_ptr = C.data_ptr<float>();

    dim3 block(BLOCK_M, BLOCK_N);
    dim3 grid((M + BLOCK_M - 1) / BLOCK_M,
              (N + BLOCK_N - 1) / BLOCK_N,
              batch);

    bmm_kernel<<<grid, block>>>(a_ptr, b_ptr, c_ptr, M, N, K);
    cudaDeviceSynchronize();
}
"""

# -------------------------------------------------------------------------
# C++ source – PYBIND11 binding
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void fused_op(torch::Tensor A, torch::Tensor B, torch::Tensor C);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op,
          "Custom batched matrix multiplication kernel");
}
"""

# -------------------------------------------------------------------------
# Compile the inline CUDA extension
# -------------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# -------------------------------------------------------------------------
# The function that will be imported for evaluation
# -------------------------------------------------------------------------
def functional_model(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Batched matrix multiplication A @ B using a hand-tuned CUDA kernel.
    Inputs are moved to the GPU if they are not already there.
    """
    # Move tensors to CUDA device if they reside on the CPU
    if A.device.type != 'cuda':
        A = A.cuda()
    if B.device.type != 'cuda':
        B = B.cuda()

    # Allocate output tensor on the same device
    batch = A.size(0)
    M = A.size(1)
    K = A.size(2)
    N = B.size(2)
    C = torch.empty((batch, M, N), dtype=torch.float32, device=A.device)

    # Launch the custom tiled kernel
    fused_ext.fused_op(A, B, C)

    return C


# -------------------------------------------------------------------------
# Helper functions for testing
# -------------------------------------------------------------------------
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
