# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_091825/code_6.py
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
# high_perf_bmm.py
# --------------------------------------------------------------
import torch
from torch.utils.cpp_extension import load_inline

# ------------------------------------------------------------------
# 1. CUDA kernel 
# ------------------------------------------------------------------
# We use a TILE structure to ensure coalesced memory access to matrix B
# and reuse of entries from matrix A in registers.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

template <int TILE_M, int TILE_N>
__global__ void batched_gemm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int batch,
    const int M,
    const int K,
    const int N)
{
    // Grid-stride over batch dimension, but for simplicity here we map 
    // each thread block to a specific batch + tile.
    int b = blockIdx.x;
    if (b >= batch) return;

    const int row = blockIdx.y * TILE_M + threadIdx.y;
    const int col = blockIdx.z * TILE_N + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        const float* A_base = A + b * M * K + row * K;
        const float* B_base = B + b * K * N + col;

        #pragma unroll
        for (int p = 0; p < K; ++p) {
            sum += A_base[p] * B_base[p * N];
        }
        C[b * M * N + row * N + col] = sum;
    }
}

void batched_gemm_cuda(at::Tensor A, at::Tensor B, at::Tensor C)
{
    const int batch = A.size(0);
    const int M = A.size(1);
    const int K = A.size(2);
    const int N = B.size(2);

    constexpr int TILE_M = 16;
    constexpr int TILE_N = 16;

    dim3 block(TILE_N, TILE_M);
    dim3 grid(batch, (M + TILE_M - 1) / TILE_M, (N + TILE_N - 1) / TILE_N);

    batched_gemm_kernel<TILE_M, TILE_N><<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        batch, M, K, N
    );
}
"""

# ------------------------------------------------------------------
# 2. C++ Binding Logic
# ------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void batched_gemm_cuda(at::Tensor A, at::Tensor B, at::Tensor C);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("batched_gemm", &batched_gemm_cuda, "Batched GEMM CUDA kernel");
}
"""

# ------------------------------------------------------------------
# 3. Compile the extension
# ------------------------------------------------------------------
fused_ext = load_inline(
    name='batched_gemm',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# ------------------------------------------------------------------
# 4. Functional Model implementation
# ------------------------------------------------------------------
def functional_model(A, B):
    # Ensure contiguous tensors
    A = A.contiguous()
    B = B.contiguous()
    
    batch, m, k = A.shape
    _, k_in, n = B.shape
    
    C = torch.empty(batch, m, n, device=A.device, dtype=A.dtype)
    
    fused_ext.batched_gemm(A, B, C)
    return C

# Constants and helpers as per requirements
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
