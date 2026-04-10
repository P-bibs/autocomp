# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_092316/code_7.py
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
# CUDA Kernel implementation
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

__global__ void bmm_kernel(const float* __restrict__ A,
                           const float* __restrict__ B,
                           float* __restrict__ C,
                           const int64_t batch,
                           const int64_t M,
                           const int64_t K,
                           const int64_t N)
{
    int b = blockIdx.z;
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;
    const float* A_batch = A + b * M * K;
    const float* B_batch = B + b * K * N;
    float* C_batch = C + b * M * N;

    for (int k_idx = 0; k_idx < (K + TILE_SIZE - 1) / TILE_SIZE; ++k_idx) {
        // Load tile into shared memory
        if (row < M && (k_idx * TILE_SIZE + tx) < K)
            sA[ty][tx] = A_batch[row * K + k_idx * TILE_SIZE + tx];
        else
            sA[ty][tx] = 0.0f;

        if ((k_idx * TILE_SIZE + ty) < K && col < N)
            sB[ty][tx] = B_batch[(k_idx * TILE_SIZE + ty) * N + col];
        else
            sB[ty][tx] = 0.0f;

        __syncthreads();

        // Accumulate result
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += sA[ty][k] * sB[k][tx];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C_batch[row * N + col] = sum;
    }
}

void bmm_forward(const at::Tensor& A, const at::Tensor& B, at::Tensor& C) {
    const int64_t batch = A.size(0);
    const int64_t M     = A.size(1);
    const int64_t K     = A.size(2);
    const int64_t N     = B.size(2);

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
              (M + TILE_SIZE - 1) / TILE_SIZE,
              batch);

    bmm_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        batch, M, K, N);
}
"""

cpp_source = r"""
#include <torch/extension.h>
void bmm_forward(const at::Tensor& A, const at::Tensor& B, at::Tensor& C);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bmm", &bmm_forward, "Custom Batched Matmul");
}
"""

# Compile the extension once
fused_ext = load_inline(
    name='fused_bmm',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3'],
    with_cuda=True
)

def functional_model(A, B):
    """
    Performance-optimized batched matrix multiplication.
    """
    A = A.contiguous().cuda()
    B = B.contiguous().cuda()
    
    batch, m, k = A.shape
    n = B.shape[2]
    
    C = torch.empty((batch, m, n), device=A.device, dtype=A.dtype)
    
    fused_ext.bmm(A, B, C)
    return C

def get_init_inputs():
    return []

def get_inputs():
    batch_size, m, k, n = 128, 512, 1024, 2048
    A = torch.rand(batch_size, m, k)
    B = torch.rand(batch_size, k, n)
    return [A, B]
