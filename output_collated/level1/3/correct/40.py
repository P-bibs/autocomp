# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_093329/code_7.py
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
# CUDA source – Tiled Batched GEMM for Performance
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

__global__ void bmm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N,
    int strideA, int strideB, int strideC)
{
    const int batch = blockIdx.z;
    const int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    const int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    float sum = 0.0f;
    const int tilesK = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;

    const float* A_batch = A + batch * strideA;
    const float* B_batch = B + batch * strideB;
    float* C_batch = C + batch * strideC;

    for (int t = 0; t < tilesK; ++t) {
        // Coalesced load into shared memory
        int aCol = t * BLOCK_SIZE + threadIdx.x;
        if (row < M && aCol < K) {
            As[threadIdx.y][threadIdx.x] = __ldg(&A_batch[row * K + aCol]);
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        int bRow = t * BLOCK_SIZE + threadIdx.y;
        if (bRow < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = __ldg(&B_batch[bRow * N + col]);
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C_batch[row * N + col] = sum;
    }
}

void bmm_cuda(const at::Tensor A, const at::Tensor B, at::Tensor C) {
    const int batch = A.size(0);
    const int M = A.size(1);
    const int K = A.size(2);
    const int N = B.size(2);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE, batch);

    bmm_kernel<<<grid, block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
        M, K, N, M * K, K * N, M * N
    );
}
"""

cpp_source = r"""
void bmm_cuda(const at::Tensor A, const at::Tensor B, at::Tensor C);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bmm_cuda", &bmm_cuda, "Custom tiled batched GEMM");
}
"""

# Compile extension
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(A, B):
    # Ensure contiguous for coalesced access
    A = A.contiguous()
    B = B.contiguous()
    batch, M, K = A.shape
    N = B.shape[2]
    C = torch.empty((batch, M, N), device=A.device, dtype=A.dtype)
    fused_ext.bmm_cuda(A, B, C)
    return C

def get_init_inputs():
    return []

def get_inputs():
    batch_size, m, k, n = 128, 512, 1024, 2048
    A = torch.rand(batch_size, m, k, device='cuda', dtype=torch.float32)
    B = torch.rand(batch_size, k, n, device='cuda', dtype=torch.float32)
    return [A, B]
