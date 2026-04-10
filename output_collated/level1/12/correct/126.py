# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_105940/code_23.py
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

# --- CUDA Kernel ---
# We optimize the memory pattern by ensuring global memory coalescing on B and out.
# Since B is (N, M), index (i, j) maps exactly to idx = i * M + j.
# A is indexed by i = idx / M.
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void broadcast_mul_kernel(const float* A, const float* B, float* out, int N, int M) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < (long long)N * M) {
        int i = idx / M;
        // B and out are accessed colaesced: idx increases linearly
        out[idx] = A[i] * B[idx];
    }
}

void fused_op_forward(torch::Tensor A, torch::Tensor B, torch::Tensor out) {
    int N = A.size(0);
    int M = B.size(1);
    long long total_elements = (long long)N * M;
    
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    
    broadcast_mul_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(), 
        B.data_ptr<float>(), 
        out.data_ptr<float>(), 
        N, M
    );
}
"""

# --- C++ Logic (Interface/Bindings) ---
cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(torch::Tensor A, torch::Tensor B, torch::Tensor out);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused broadcast multiplication");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(A, B):
    """
    Optimized broadcast multiplication using custom CUDA kernel.
    Input A: shape (N,)
    Input B: shape (N, M)
    Output: shape (N, M)
    """
    N = A.size(0)
    M = B.size(1)
    # Output tensor pre-allocated for the kernel
    out = torch.empty((N, M), device=A.device, dtype=A.dtype)
    
    fused_ext.fused_op(A, B, out)
    return out

# --- Verification & Setup ---
M = 4096
N = 4096

def get_init_inputs():
    return []

def get_inputs():
    # Use GPU tensors as requested for performance
    A = torch.rand(N, device='cuda')
    B = torch.rand(N, M, device='cuda')
    return [A, B]
