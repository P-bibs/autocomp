# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_101548/code_7.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = []
FORWARD_ARG_NAMES = ['A', 'B']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = []
REQUIRED_FLAT_STATE_NAMES = []


class ModelNew(nn.Module):
    """
    Simple model that performs a matrix multiplication of a diagonal matrix with another matrix.
    C = diag(A) * B
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

# ------------------------------------------------------------
# 1. Custom CUDA Kernel Implementation
# ------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 8

__global__ void mul_kernel(const float* __restrict__ A,
                           const float* __restrict__ B,
                           float* __restrict__ C,
                           int N, int M)
{
    // Shared memory to store the scalar A[row] for this block's rows
    // Only BLOCK_SIZE_Y floats needed since one row uses one scalar
    __shared__ float sA[BLOCK_SIZE_Y];

    int row = blockIdx.y * BLOCK_SIZE_Y + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE_X + threadIdx.x;

    // Load A[row] into shared memory once per row per block
    if (threadIdx.x == 0 && row < N) {
        sA[threadIdx.y] = A[row];
    }
    __syncthreads();

    // Perform element-wise multiplication
    if (row < N && col < M) {
        C[row * M + col] = sA[threadIdx.y] * B[row * M + col];
    }
}

void fused_op_forward(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    int N = A.size(0);
    int M = B.size(1);

    dim3 blocks((M + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, 
                (N + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);
    dim3 threads(BLOCK_SIZE_X, BLOCK_SIZE_Y);

    mul_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N, M);
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(torch::Tensor A, torch::Tensor B, torch::Tensor C);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Tile-wise A*B multiplication");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_mul_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# ------------------------------------------------------------
# 2. Optimized functional_model
# ------------------------------------------------------------
def functional_model(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Optimized broadcast multiplication using custom CUDA kernel.
    """
    if not A.is_cuda: A = A.cuda()
    if not B.is_cuda: B = B.cuda()
    
    C = torch.empty_like(B)
    fused_ext.fused_op(A, B, C)
    return C

# ------------------------------------------------------------
# 3. Required Helper Functions (as per instructions)
# ------------------------------------------------------------
N, M = 4096, 4096

def get_init_inputs():
    return []

def get_inputs():
    A = torch.rand(N, device='cuda', dtype=torch.float32)
    B = torch.rand(N, M, device='cuda', dtype=torch.float32)
    return [A, B]
