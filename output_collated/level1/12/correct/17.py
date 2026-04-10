# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_101900/code_12.py
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

# Custom CUDA kernel for fused unsqueeze and multiply operation
# Each thread calculates one row-slice or element to maximize coalesced access on B and C.
# Since B is (N, M), row-major, and C is (N, M), accessing these linearly is perfectly coalesced.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_unsqueeze_multiply_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int N,
    const int M
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * M;
    
    // Grid-stride loop for scalability
    for (int i = idx; i < total_elements; i += blockDim.x * gridDim.x) {
        // i / M gives the row index 'n'.
        // For every element in row 'n', we multiply by A[n].
        // This is highly efficient as A[n] stays in register or cache 
        // while B and C are accessed linearly (coalesced).
        int row = i / M;
        C[i] = A[row] * B[i];
    }
}

void fused_unsqueeze_multiply_forward(
    const torch::Tensor& A,
    const torch::Tensor& B,
    torch::Tensor& C
) {
    int N = A.size(0);
    int M = B.size(1);
    int total_elements = N * M;
    
    const int threads_per_block = 256;
    const int blocks_per_grid = (total_elements + threads_per_block - 1) / threads_per_block;
    
    fused_unsqueeze_multiply_kernel<<<blocks_per_grid, threads_per_block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N,
        M
    );
}
"""

# C++ binding code
cpp_source = r"""
#include <torch/extension.h>

void fused_unsqueeze_multiply_forward(
    const torch::Tensor& A,
    const torch::Tensor& B,
    torch::Tensor& C
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_unsqueeze_multiply", &fused_unsqueeze_multiply_forward, "Fused multiply kernel");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_unsqueeze_multiply',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(A, B):
    """
    Optimized implementation of A.unsqueeze(1) * B using a fused CUDA kernel.
    """
    N = A.shape[0]
    M = B.shape[1]
    
    # Ensure inputs are contiguous on GPU
    if not A.is_cuda: A = A.cuda()
    if not B.is_cuda: B = B.cuda()
    if not A.is_contiguous(): A = A.contiguous()
    if not B.is_contiguous(): B = B.contiguous()
    
    # Allocate output tensor
    C = torch.empty((N, M), dtype=torch.float32, device='cuda')
    
    # Call custom CUDA kernel
    fused_ext.fused_unsqueeze_multiply(A, B, C)
    
    return C

N = 4096
M = 4096

def get_init_inputs():
    return []

def get_inputs():
    A = torch.rand(N)
    B = torch.rand(N, M)
    return [A, B]
