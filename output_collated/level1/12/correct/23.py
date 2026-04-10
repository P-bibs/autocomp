# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_101900/code_18.py
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

# ----------------------------------------------------------------------
# CUDA kernel that performs out[i, j] = A[i] * B[i, j] using shared memory
# ----------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void mul_shared_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ Out,
    const int N,
    const int M)
{
    // Shared memory stores one A value per row of the thread block
    extern __shared__ float sA[];

    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Load A[row] into shared memory once per row in the block
    if (threadIdx.x == 0 && row < N) {
        sA[threadIdx.y] = A[row];
    }
    __syncthreads();

    // Compute: each thread multiplies the cached scalar with local B element
    if (row < N && col < M) {
        Out[row * M + col] = sA[threadIdx.y] * B[row * M + col];
    }
}

void mul_shared_launcher(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor Out)
{
    const int N = A.size(0);
    const int M = B.size(1);

    // Grid dimensions: 32 threads in X (coalesced B read), 8 in Y (row reuse)
    // 256 threads per block
    const int tx = 32;
    const int ty = 8;
    dim3 block(tx, ty);
    dim3 grid((M + tx - 1) / tx, (N + ty - 1) / ty);

    // Shared memory size = ty * sizeof(float)
    // A.data_ptr is accessed directly as float pointer
    mul_shared_kernel<<<grid, block, ty * sizeof(float)>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        Out.data_ptr<float>(),
        N,
        M
    );
}
"""

# ----------------------------------------------------------------------
# C++ Bindings
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void mul_shared_launcher(torch::Tensor A, torch::Tensor B, torch::Tensor Out);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mul_shared", &mul_shared_launcher, "Row-wise scalar multiply with shared memory");
}
"""

# Build the extension
_ext = load_inline(
    name='mul_shared_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Optimized implementation using shared memory to cache A[i].
    """
    # Ensure contiguous for coalesced access
    A = A.contiguous()
    B = B.contiguous()
    
    out = torch.empty_like(B)
    _ext.mul_shared(A, B, out)
    return out

# ----------------------------------------------------------------------
# Input generators
# ----------------------------------------------------------------------
M = 4096
N = 4096

def get_init_inputs():
    return []

def get_inputs():
    A = torch.rand(N, device='cuda')
    B = torch.rand(N, M, device='cuda')
    return [A, B]
