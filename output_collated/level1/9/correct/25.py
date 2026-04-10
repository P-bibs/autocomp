# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_100346/code_7.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = []
FORWARD_ARG_NAMES = ['A', 'B']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = []
REQUIRED_FLAT_STATE_NAMES = []


class ModelNew(nn.Module):
    """
    Simple model that performs a single matrix multiplication (C = A * B) where one of the matrices is tall and skinny (M >> N or N >> M)
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
# CUDA source – Optimized Tiled Matrix Multiplier
# M = 32768, N = 32
# Strategy:
# 1. Block size is 32x32.
# 2. Each block computes one 32x32 tile of the output matrix C.
# 3. Given N=32, the loop over the K-dimension (N) runs only once (numTiles=1).
# 4. This matches the memory access pattern: A is (M, N), B is (N, M).
#    C[row, col] = sum(A[row, k] * B[k, col]) for k = 0..31
# -------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

__global__ void tiled_matmul_kernel(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float*       __restrict__ C,
                                    int M, int N)
{
    // Shared memory for tiles (32x32 floats = 4KB per tile, 8KB total)
    // This allows coalesced access and data reuse.
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.x * TILE_SIZE + threadIdx.x;
    int col = blockIdx.y * TILE_SIZE + threadIdx.y;

    float sum = 0.0f;

    // K-dimension is N=32, so we loop through the internal dimension
    // Since TILE_SIZE is 32 and N is 32, this effectively processes in one iteration
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        int aCol = t * TILE_SIZE + threadIdx.y;
        if (row < M && aCol < N)
            As[threadIdx.x][threadIdx.y] = A[row * N + aCol];
        else
            As[threadIdx.x][threadIdx.y] = 0.0f;

        int bRow = t * TILE_SIZE + threadIdx.x;
        if (bRow < N && col < M)
            Bs[threadIdx.x][threadIdx.y] = B[bRow * M + col];
        else
            Bs[threadIdx.x][threadIdx.y] = 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[threadIdx.x][k] * Bs[k][threadIdx.y];
        }
        __syncthreads();
    }

    if (row < M && col < M) {
        C[row * M + col] = sum;
    }
}

void tiled_matmul(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& C) {
    const int M = A.size(0);
    const int N = A.size(1);

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 grid((M + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    tiled_matmul_kernel<<<grid, threads>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void tiled_matmul(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& C);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tiled_matmul", &tiled_matmul, "Shared memory tiled matmul");
}
"""

# Compile the extension inline
matmul_ext = load_inline(
    name='matmul_tiled',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(A, B):
    # Ensure inputs are contiguous and on device
    A = A.contiguous().cuda()
    B = B.contiguous().cuda()
    
    M = A.size(0)
    C = torch.empty((M, M), dtype=A.dtype, device=A.device)
    
    matmul_ext.tiled_matmul(A, B, C)
    
    return C
