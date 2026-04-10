# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_094844/code_0.py
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

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_DIM 32

__global__ void tiled_matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N
) {
    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * TILE_DIM + ty;
    int col = bx * TILE_DIM + tx;

    float sum = 0.0f;

    for (int k = 0; k < (N + TILE_DIM - 1) / TILE_DIM; ++k) {
        // Load A tile
        if (row < M && k * TILE_DIM + tx < N)
            As[ty][tx] = A[row * N + k * TILE_DIM + tx];
        else
            As[ty][tx] = 0.0f;

        // Load B tile
        if (k * TILE_DIM + ty < N && col < M)
            Bs[ty][tx] = B[(k * TILE_DIM + ty) * M + col];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_DIM; ++i) {
            sum += As[ty][i] * Bs[i][tx];
        }

        __syncthreads();
    }

    if (row < M && col < M)
        C[row * M + col] = sum;
}

void tiled_matmul_forward(int blocks_x, int blocks_y, int threads_x, int threads_y, 
                          const float* A, const float* B, float* C, int M, int N) {
    dim3 grid(blocks_x, blocks_y);
    dim3 block(threads_x, threads_y);
    tiled_matmul_kernel<<<grid, block>>>(A, B, C, M, N);
}
"""

cpp_source = r"""
#include <torch/extension.h>

void tiled_matmul_forward(int blocks_x, int blocks_y, int threads_x, int threads_y,
                          const float* A, const float* B, float* C, int M, int N);

void launch_tiled_matmul(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    int M = A.size(0);
    int N = A.size(1);

    const float* A_ptr = A.data_ptr<float>();
    const float* B_ptr = B.data_ptr<float>();
    float* C_ptr = C.data_ptr<float>();

    int TILE_DIM = 32;
    int blocks_x = (M + TILE_DIM - 1) / TILE_DIM;
    int blocks_y = (M + TILE_DIM - 1) / TILE_DIM;
    int threads_x = TILE_DIM;
    int threads_y = TILE_DIM;

    tiled_matmul_forward(blocks_x, blocks_y, threads_x, threads_y, A_ptr, B_ptr, C_ptr, M, N);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tiled_mm", &launch_tiled_matmul, "Tiled Matrix Multiplication");
}
"""

# Compile the extension
tiled_mm_ext = load_inline(
    name='tiled_mm',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(A, B):
    M = A.shape[0]
    device = A.device
    dtype = A.dtype

    # Allocate output tensor
    C = torch.empty((M, M), dtype=dtype, device=device)
    
    # Ensure inputs are contiguous
    A = A.contiguous()
    B = B.contiguous()
    
    # Call custom CUDA kernel
    tiled_mm_ext.tiled_mm(A, B, C)

    return C

# Test inputs setup
M_val = 16384 * 2
N_val = 16 * 2

def get_init_inputs():
    return []

def get_inputs():
    A = torch.rand(M_val, N_val).cuda()
    B = torch.rand(N_val, M_val).cuda()
    return [A, B]
