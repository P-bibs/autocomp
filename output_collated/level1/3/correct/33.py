# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_092739/code_5.py
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

# CUDA kernel with Tiled Matrix Multiplication
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void matmul_tiled_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int K, int N) {
    int batch_idx = blockIdx.z;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    const float* A_batch = A + (size_t)batch_idx * M * K;
    const float* B_batch = B + (size_t)batch_idx * K * N;
    float* C_batch = C + (size_t)batch_idx * M * N;

    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    float val = 0.0f;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    for (int k = 0; k < (K + TILE_SIZE - 1) / TILE_SIZE; ++k) {
        // Load A tile
        if (row < M && (k * TILE_SIZE + tx) < K) 
            sA[ty][tx] = A_batch[row * K + (k * TILE_SIZE + tx)];
        else 
            sA[ty][tx] = 0.0f;

        // Load B tile
        if (col < N && (k * TILE_SIZE + ty) < K) 
            sB[ty][tx] = B_batch[(k * TILE_SIZE + ty) * N + col];
        else 
            sB[ty][tx] = 0.0f;

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_SIZE; ++i) {
            val += sA[ty][i] * sB[i][tx];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C_batch[row * N + col] = val;
    }
}

torch::Tensor matmul_tiled(torch::Tensor A, torch::Tensor B) {
    int batch = A.size(0);
    int M = A.size(1);
    int K = A.size(2);
    int N = B.size(2);
    
    auto C = torch::zeros({batch, M, N}, A.options());

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE, batch);

    matmul_tiled_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N);
    
    return C;
}
"""

cpp_source = r"""
#include <torch/extension.h>
torch::Tensor matmul_tiled(torch::Tensor A, torch::Tensor B);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_tiled", &matmul_tiled, "Tiled Batch Matrix Multiplication");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='matmul_tiled_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(A, B):
    # Ensure inputs are on GPU and contiguous to match kernel expectations
    return fused_ext.matmul_tiled(A.contiguous().cuda(), B.contiguous().cuda())

# Configuration consistent with original requirement
batch_size = 128
m = 128 * 4
k = 256 * 4
n = 512 * 4

def get_inputs():
    A = torch.rand(batch_size, m, k, device='cuda')
    B = torch.rand(batch_size, k, n, device='cuda')
    return [A, B]
