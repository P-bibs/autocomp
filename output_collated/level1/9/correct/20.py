# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_100346/code_2.py
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# --- Compile the custom CUDA kernel ---
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

__global__ void gemm_shared_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M,
    int N
) {
    // Shared memory for tiles
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Block row and column
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float acc = 0.0f;

    // Load tiles into shared memory
    if (row < M && col < M) {
        As[ty][tx] = A[row * N + tx];
        Bs[ty][tx] = B[ty * M + col];
    } else {
        As[ty][tx] = 0.0f;
        Bs[ty][tx] = 0.0f;
    }

    __syncthreads();

    // Compute dot product
    for (int k = 0; k < TILE_SIZE; ++k) {
        acc += As[ty][k] * Bs[k][tx];
    }

    // Write result
    if (row < M && col < M) {
        C[row * M + col] = acc;
    }
}

void gemm_shared_forward(
    const torch::Tensor& A,
    const torch::Tensor& B,
    torch::Tensor& C,
    int M,
    int N
) {
    dim3 gridDim((M + 31) / 32, (M + 31) / 32);
    dim3 blockDim(32, 32);

    gemm_shared_kernel<<<gridDim, blockDim>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M,
        N
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void gemm_shared_forward(
    const torch::Tensor& A,
    const torch::Tensor& B,
    torch::Tensor& C,
    int M,
    int N
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_shared", &gemm_shared_forward, "GEMM with shared memory tiling");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

M = 16384 * 2
N = 16 * 2

def get_init_inputs():
    return []

def get_inputs():
    A = torch.rand(M, N).cuda()
    B = torch.rand(N, M).cuda()
    return [A, B]

def functional_model(A, B):
    # A: (M, N), B: (N, M)
    C = torch.empty(M, M, device='cuda', dtype=torch.float32)
    fused_ext.gemm_shared(A, B, C, M, N)
    return C
