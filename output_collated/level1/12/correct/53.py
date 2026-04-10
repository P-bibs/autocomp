# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_102701/code_31.py
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
# Optimized CUDA kernel
# ----------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// Vectorized loading and shared memory caching of the A vector
// Optimization: A is loaded once per block into shared memory.
// Row and column indices are computed using block/thread IDs to avoid division.
__global__ void fused_unsqueeze_multiply_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N,
    int M
) {
    extern __shared__ float s_A[];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Load A[row] into shared memory once per row
    if (threadIdx.x == 0 && row < N) {
        s_A[threadIdx.y] = A[row];
    }
    __syncthreads();

    if (row < N && col < M) {
        float a_val = s_A[threadIdx.y];
        // Use __ldg to hint read-only cache for B
        float b_val = __ldg(&B[row * M + col]);
        C[row * M + col] = a_val * b_val;
    }
}

void fused_unsqueeze_multiply_forward(
    const torch::Tensor& A,
    const torch::Tensor& B,
    torch::Tensor& C,
    int N,
    int M
) {
    // 32x8 block size provides 256 threads and good occupancy
    const int block_x = 32;
    const int block_y = 8;
    dim3 block(block_x, block_y);
    dim3 grid((M + block_x - 1) / block_x, (N + block_y - 1) / block_y);

    size_t shared_mem_size = block_y * sizeof(float);

    fused_unsqueeze_multiply_kernel<<<grid, block, shared_mem_size>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N,
        M
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_unsqueeze_multiply_forward(
    const torch::Tensor& A,
    const torch::Tensor& B,
    torch::Tensor& C,
    int N,
    int M
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_unsqueeze_multiply", &fused_unsqueeze_multiply_forward, "Fused unsqueeze and multiply");
}
"""

# Compile the extension once
fused_ext = load_inline(
    name='fused_unsqueeze_multiply',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(A, B):
    """
    Optimized implementation of A.unsqueeze(-1) * B.
    Equivalent to broadcasting A across the columns of B.
    """
    N = A.size(0)
    M = B.size(1)
    
    # Ensure contiguous memory for kernel access
    if not A.is_contiguous(): A = A.contiguous()
    if not B.is_contiguous(): B = B.contiguous()
    
    C = torch.empty(N, M, dtype=torch.float32, device='cuda')
    
    fused_ext.fused_unsqueeze_multiply(A.cuda(), B.cuda(), C, N, M)
    
    return C

def get_init_inputs():
    return []

def get_inputs():
    # Use standard sizes defined in the problem
    N, M = 4096, 4096
    A = torch.rand(N).cuda()
    B = torch.rand(N, M).cuda()
    return [A, B]
