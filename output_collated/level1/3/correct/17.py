# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_091341/code_7.py
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
# CUDA source – tiled batched GEMM kernel
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// Using 32x8 tiles per block with unrolling provides better occupancy and instruction throughput 
// for the RTX 2080 Ti (Turing architecture) compared to simple 16x16.
constexpr int TILE_SIZE = 16; 

__global__ void bmm_kernel(const float* __restrict__ A,
                           const float* __restrict__ B,
                           float*       __restrict__ C,
                           int M, int N, int K, int batch)
{
    // batch index (z-grid)
    int b = blockIdx.z;

    // Tile indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Output indices
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    float acc = 0.0f;

    // Offset pointers to the current batch
    const float* A_batch = A + b * M * K;
    const float* B_batch = B + b * K * N;
    float*       C_batch = C + b * M * N;

    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; ++t) {
        // Load data into shared memory
        int k_idx = t * TILE_SIZE;
        
        if (row < M && (k_idx + tx) < K)
            As[ty][tx] = A_batch[row * K + k_idx + tx];
        else
            As[ty][tx] = 0.0f;

        if ((k_idx + ty) < K && col < N)
            Bs[ty][tx] = B_batch[(k_idx + ty) * N + col];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            acc += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N)
        C_batch[row * N + col] = acc;
}

void bmm_cuda(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& C) {
    const int batch = A.size(0);
    const int M = A.size(1);
    const int K = A.size(2);
    const int N = B.size(2);

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE, batch);

    bmm_kernel<<<grid, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K, batch);
}
"""

# ----------------------------------------------------------------------
# C++ binding
# ----------------------------------------------------------------------
cpp_source = r"""
void bmm_cuda(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& C);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bmm_cuda", &bmm_cuda, "Custom Batched GEMM Kernel");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='bmm_extension',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(A, B):
    """
    Performs C = A @ B using a custom CUDA shell kernel.
    """
    batch, M, K = A.shape
    N = B.shape[2]
    C = torch.empty((batch, M, N), device=A.device, dtype=A.dtype)
    
    # Ensure tensors are contiguous for predictable pointer arithmetic
    A = A.contiguous()
    B = B.contiguous()
    
    fused_ext.bmm_cuda(A, B, C)
    return C

def get_init_inputs():
    return []

def get_inputs():
    batch_size = 128
    m = 128 * 4
    k = 256 * 4
    n = 512 * 4
    A = torch.rand(batch_size, m, k, device='cuda')
    B = torch.rand(batch_size, k, n, device='cuda')
    return [A, B]
