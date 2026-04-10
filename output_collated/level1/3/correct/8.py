# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_090902/code_4.py
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

# The CUDA kernel implements tiling with a 16x16 tile size. 
# Shared memory reduces global memory bandwidth bottleneck by reusing loaded data.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void batch_matmul_tiled_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int batch_size, int m, int n, int k
) {
    int bx = blockIdx.x; int by = blockIdx.y; int bz = blockIdx.z;
    int tx = threadIdx.x; int ty = threadIdx.y;

    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;

    for (int t = 0; t < (k + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        if (row < m && (t * TILE_SIZE + tx) < k)
            As[ty][tx] = A[bz * m * k + row * k + t * TILE_SIZE + tx];
        else
            As[ty][tx] = 0.0f;

        if (col < n && (t * TILE_SIZE + ty) < k)
            Bs[ty][tx] = B[bz * k * n + (t * TILE_SIZE + ty) * n + col];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_SIZE; ++i) {
            sum += As[ty][i] * Bs[i][tx];
        }
        __syncthreads();
    }

    if (row < m && col < n) {
        C[bz * m * n + row * n + col] = sum;
    }
}

void batch_matmul_tiled_launcher(
    const float* A, const float* B, float* C,
    int batch_size, int m, int n, int k
) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((n + TILE_SIZE - 1) / TILE_SIZE, (m + TILE_SIZE - 1) / TILE_SIZE, batch_size);
    batch_matmul_tiled_kernel<<<grid, block>>>(A, B, C, batch_size, m, n, k);
}
"""

cpp_source = r"""
#include <torch/extension.h>

void batch_matmul_tiled_launcher(const float* A, const float* B, float* C, int batch_size, int m, int n, int k);

torch::Tensor batch_matmul(torch::Tensor A, torch::Tensor B) {
    int batch_size = A.size(0);
    int m = A.size(1);
    int k = A.size(2);
    int n = B.size(2);
    auto C = torch::zeros({batch_size, m, n}, A.options());
    batch_matmul_tiled_launcher(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), batch_size, m, n, k);
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("batch_matmul", &batch_matmul, "Tiled batch matmul");
}
"""

# Compile as a JIT-compiled C++ extension
tiled_ext = load_inline(
    name='tiled_bmm_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(A, B):
    return tiled_ext.batch_matmul(A, B)

batch_size, m, k, n = 128, 512, 1024, 2048

def get_init_inputs():
    return []

def get_inputs():
    return [torch.rand(batch_size, m, k).cuda(), torch.rand(batch_size, k, n).cuda()]
