# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_101900/code_2.py
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

# Optimization: Grid-Stride Loops (Optimization #3)
# We restructure the kernel to use grid-stride loops which improves:
# 1. GPU occupancy by using optimal block sizes (1024 threads per block)
# 2. Scalability across different problem sizes
# 3. Better load balancing by having each thread process multiple elements
# 4. Reduced kernel launch overhead

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_multiply_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ Out, int N, int M) {
    // Grid-stride loop pattern: each thread processes multiple elements
    int total_elements = N * M;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Each thread processes multiple elements in a grid-stride pattern
    for (int i = tid; i < total_elements; i += stride) {
        int row = i / M;  // Compute row index
        int col = i % M;  // Compute column index
        Out[i] = A[row] * B[i];  // Perform fused multiply
    }
}

void fused_op_forward(torch::Tensor A, torch::Tensor B, torch::Tensor Out) {
    int N = A.size(0);
    int M = B.size(1);
    int total_elements = N * M;
    
    // Use optimal block size for better occupancy (1024 threads per block)
    // This is more efficient than the previous (32, 16) = 512 threads
    int threads_per_block = 1024;
    
    // Calculate number of blocks needed, capped to avoid oversized grids
    int blocks = min(65535, (total_elements + threads_per_block - 1) / threads_per_block);
    
    fused_multiply_kernel<<<blocks, threads_per_block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), Out.data_ptr<float>(), N, M);
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(torch::Tensor A, torch::Tensor B, torch::Tensor Out);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused broadcasting multiply with grid-stride loops");
}
"""

# Compile the extension inline
fused_ext = load_inline(
    name='fused_mul',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(A, B):
    N = A.shape[0]
    M = B.shape[1]
    out = torch.empty((N, M), device=A.device, dtype=A.dtype)
    fused_ext.fused_op(A, B, out)
    return out

# --- Environment Setup (as requested by problem statement) ---
M, N = 4096, 4096

def get_init_inputs():
    return []

def get_inputs():
    # Ensure tensors are on GPU as per requirement 1
    A = torch.rand(N, device='cuda')
    B = torch.rand(N, M, device='cuda')
    return [A, B]
