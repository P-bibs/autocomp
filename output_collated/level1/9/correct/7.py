# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_094844/code_5.py
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

# Optimization: Tiled Matrix Multiplication using Shared Memory
# We use shared memory to cache blocks of A and B, significantly reducing 
# global memory latency. TILE_SIZE is set to 16 for memory efficiency 
# on the RTX 2080Ti.

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        if (row < M && (t * TILE_SIZE + tx) < K)
            sA[ty][tx] = A[row * K + (t * TILE_SIZE + tx)];
        else
            sA[ty][tx] = 0.0f;

        if ((t * TILE_SIZE + ty) < K && col < M)
            sB[ty][tx] = B[(t * TILE_SIZE + ty) * M + col];
        else
            sB[ty][tx] = 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += sA[ty][k] * sB[k][tx];
        }
        __syncthreads();
    }
    if (row < M && col < M) {
        C[row * M + col] = sum;
    }
}

void launch_matmul(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    int M = A.size(0);
    int K = A.size(1);
    
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((M + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    
    matmul_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, M, K
    );
}
"""

cpp_source = r"""
void launch_matmul(torch::Tensor A, torch::Tensor B, torch::Tensor C);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul", &launch_matmul, "Tiled Matrix Multiplication Kernel");
}
"""

# Compile the extension
matmul_ext = load_inline(
    name='matmul_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(A, B):
    """
    Computes A @ B using a custom CUDA kernel.
    Input A shape: (M, K)
    Input B shape: (K, M)
    Output shape: (M, M)
    """
    M = A.shape[0]
    output = torch.empty((M, M), device='cuda', dtype=torch.float32)
    matmul_ext.matmul(A, B, output)
    return output

M, N = 16384 * 2, 16 * 2

def get_init_inputs():
    return []

def get_inputs():
    A = torch.rand(M, N, device='cuda', dtype=torch.float32)
    B = torch.rand(N, M, device='cuda', dtype=torch.float32)
    return [A, B]
