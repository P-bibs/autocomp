# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_090902/code_7.py
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

# -----------------------------------------------------------------------------
# Compilation of the CUDA extension
# Using a 32x32 TILE_SIZE for shared memory to optimize global memory bandwidth.
# -----------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

__global__ void bmm_kernel(const float* __restrict__ A, 
                           const float* __restrict__ B, 
                           float* __restrict__ C,
                           int M, int K, int N) {
    // Shared memory for tiles
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int batch = blockIdx.x;
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.z * TILE_SIZE + tx;

    float sum = 0.0f;

    // Offset pointers to the current batch
    const float* A_batch = A + (batch * M * K);
    const float* B_batch = B + (batch * K * N);
    float* C_batch = C + (batch * M * N);

    for (int kk = 0; kk < K; kk += TILE_SIZE) {
        // Load tile A
        if (row < M && (kk + tx) < K)
            As[ty][tx] = A_batch[row * K + (kk + tx)];
        else
            As[ty][tx] = 0.0f;

        // Load tile B
        if ((kk + ty) < K && col < N)
            Bs[ty][tx] = B_batch[(kk + ty) * N + col];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();

        // Compute partial dot product
        #pragma unroll
        for (int i = 0; i < TILE_SIZE; ++i) {
            sum += As[ty][i] * Bs[i][tx];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C_batch[row * N + col] = sum;
    }
}

void bmm_cuda(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& C) {
    const int batch = A.size(0);
    const int M = A.size(1);
    const int K = A.size(2);
    const int N = B.size(2);

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid(batch, (M + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    bmm_kernel<<<grid, block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void bmm_cuda(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& C);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bmm_cuda", &bmm_cuda, "Batched Matmul CUDA");
}
"""

bmm_ext = load_inline(
    name='bmm_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# -----------------------------------------------------------------------------
# Functional model
# -----------------------------------------------------------------------------
def functional_model(A, B):
    # Ensure inputs are on GPU and contiguous
    if not A.is_cuda: A = A.cuda()
    if not B.is_cuda: B = B.cuda()
    A = A.contiguous()
    B = B.contiguous()
    
    # Initialize output buffer
    batch_size = A.size(0)
    m = A.size(1)
    n = B.size(2)
    C = torch.empty((batch_size, m, n), device='cuda', dtype=torch.float32)
    
    # Invoke custom kernel
    bmm_ext.bmm_cuda(A, B, C)
    
    return C

def get_init_inputs():
    return []

def get_inputs():
    batch_size = 128
    m, k, n = 512, 1024, 2048
    return [torch.rand(batch_size, m, k), torch.rand(batch_size, k, n)]
