# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_103853/code_19.py
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

# The CUDA kernel performs the operation output[n, m] = A[n] * B[n, m].
# Because the memory is stored in row-major order (standard for PyTorch),
# index i = n * M + m. Since m changes fastest, i is contiguous in memory.
# Global memory accesses are coalesced because consecutive threads access 
# consecutive indices i, which map to consecutive memory locations.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_broadcast_mul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ output,
    const int N,
    const int M
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * M;
    
    if (idx < total_elements) {
        // Since the matrix is M-wide, row index n = idx / M.
        // A[n] is broadcasted across the row.
        int n = idx / M;
        output[idx] = A[n] * B[idx];
    }
}

void fused_broadcast_mul_forward(torch::Tensor A, torch::Tensor B, torch::Tensor output) {
    const int N = A.size(0);
    const int M = B.size(1);
    const int total_elements = N * M;
    
    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;
    
    fused_broadcast_mul_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(), 
        B.data_ptr<float>(), 
        output.data_ptr<float>(), 
        N, M
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_broadcast_mul_forward(torch::Tensor A, torch::Tensor B, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_broadcast_mul", &fused_broadcast_mul_forward, "Fused broadcast multiplication");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_broadcast_mul_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(A, B):
    # Ensure inputs are on correct device and type
    A = A.to(device='cuda', dtype=torch.float32)
    B = B.to(device='cuda', dtype=torch.float32)
    
    # Create empty output tensor
    output = torch.empty_like(B)
    
    # Run custom kernel
    fused_ext.fused_broadcast_mul(A, B, output)
    
    return output

M = 4096
N = 4096

def get_init_inputs():
    return []

def get_inputs():
    # Helper to generate inputs for evaluation
    A = torch.rand(N)
    B = torch.rand(N, M)
    return [A, B]
