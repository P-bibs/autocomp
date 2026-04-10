# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_101548/code_4.py
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

# The CUDA kernel uses a 1D grid for coalesced memory access.
# Since output[n, m] = A[n] * B[n, m], and M is the inner dimension,
# threads access B[idx] and output[idx] sequentially, ensuring
# coalesced memory reads and writes. A[n] (where n = idx / M) 
# is read once per row, which fits well within the L1 cache.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void broadcast_mul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ output,
    const int N,
    const int M
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * M) return;

    // Row index n determines which A[n] to load.
    // column index is implicit in the linear idx.
    int n = idx / M;
    output[idx] = A[n] * B[idx];
}

void broadcast_mul_forward(
    const torch::Tensor& A,
    const torch::Tensor& B,
    torch::Tensor& output
) {
    const int N = A.size(0);
    const int M = B.size(1);
    const int total_elements = N * M;
    
    const int threads_per_block = 256;
    const int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    broadcast_mul_kernel<<<blocks, threads_per_block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        output.data_ptr<float>(),
        N,
        M
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void broadcast_mul_forward(
    const torch::Tensor& A,
    const torch::Tensor& B,
    torch::Tensor& output
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("broadcast_mul", &broadcast_mul_forward, "Broadcast multiplication kernel");
}
"""

# Compile the extension just-in-time
fused_ext = load_inline(
    name='broadcast_mul_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(A, B):
    """
    Optimized broadcasting multiplication A.unsqueeze(1) * B.
    Handles tensors residing on GPU.
    """
    # Ensure inputs are contiguous to satisfy kernel's indexing logic
    if not A.is_contiguous(): A = A.contiguous()
    if not B.is_contiguous(): B = B.contiguous()
    
    # Create output tensor on the same device as B
    output = torch.empty_like(B)
    
    # Execute the custom fused kernel
    fused_ext.broadcast_mul(A, B, output)
    
    return output

# Parameters according to problem description
M = 4096
N = 4096

def get_init_inputs():
    return []

def get_inputs():
    # Return inputs on GPU for performance benchmarking
    A = torch.rand(N).cuda()
    B = torch.rand(N, M).cuda()
    return [A, B]
