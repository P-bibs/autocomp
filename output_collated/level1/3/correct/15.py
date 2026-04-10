# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_091341/code_4.py
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

# Tiled CUDA kernel for performance
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void batch_matmul_tiled_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int batch_size, int m, int k, int n
) {
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x, by = blockIdx.y, bz = blockIdx.z;
    int tx = threadIdx.x, ty = threadIdx.y;

    int col = bx * TILE_SIZE + tx;
    int row = by * TILE_SIZE + ty;

    float sum = 0.0f;
    int num_tiles = (k + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; ++t) {
        if (row < m && (t * TILE_SIZE + tx) < k)
            sA[ty][tx] = A[bz * m * k + row * k + t * TILE_SIZE + tx];
        else
            sA[ty][tx] = 0.0f;

        if (col < n && (t * TILE_SIZE + ty) < k)
            sB[ty][tx] = B[bz * k * n + (t * TILE_SIZE + ty) * n + col];
        else
            sB[ty][tx] = 0.0f;

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; ++i)
            sum += sA[ty][i] * sB[i][tx];

        __syncthreads();
    }

    if (row < m && col < n) {
        C[bz * m * n + row * n + col] = sum;
    }
}

void launch_batch_matmul(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    int batch = A.size(0);
    int m = A.size(1);
    int k = A.size(2);
    int n = B.size(2);

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((n + TILE_SIZE - 1) / TILE_SIZE, (m + TILE_SIZE - 1) / TILE_SIZE, batch);
    
    batch_matmul_tiled_kernel<<<grid, block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), 
        batch, m, k, n
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void launch_batch_matmul(torch::Tensor A, torch::Tensor B, torch::Tensor C);

torch::Tensor fused_batch_matmul(torch::Tensor A, torch::Tensor B) {
    auto C = torch::empty({A.size(0), A.size(1), B.size(2)}, A.options());
    launch_batch_matmul(A, B, C);
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_batch_matmul", &fused_batch_matmul, "Tiled batch matmul");
}
"""

optimized_ext = load_inline(
    name='optimized_matmul',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(A, B):
    return optimized_ext.fused_batch_matmul(A, B)

batch_size, m, k, n = 128, 512, 1024, 2048

def get_init_inputs():
    return []

def get_inputs():
    return [torch.rand(batch_size, m, k, device='cuda'), 
            torch.rand(batch_size, k, n, device='cuda')]
