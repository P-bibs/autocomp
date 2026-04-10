# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_104927/code_3.py
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_op_forward_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int N,
    const int M
) {
    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Grid-stride loop to handle cases where we have more elements than threads
    for (int i = idx; i < N * M; i += stride) {
        int row = i / M;
        int col = i % M;
        C[i] = A[row] * B[i];
    }
}

void fused_op_forward(int blocks, int threads, 
                      const float* A, const float* B, float* C,
                      const int N, const int M) {
    fused_op_forward_kernel<<<blocks, threads>>>(A, B, C, N, M);
}
"""

# --- C++ Logic (Interface/Bindings) ---
cpp_source = r"""
#include <torch/extension.h>

// Forward declaration of the function in the .cu file
void fused_op_forward(int blocks, int threads,
                      const float* A, const float* B, float* C,
                      const int N, const int M);

void fused_op(at::Tensor A, at::Tensor B, at::Tensor C) {
    const int N = A.size(0);
    const int M = B.size(1);
    
    // Launch configuration
    const int threads_per_block = 256;
    const int blocks = (N * M + threads_per_block - 1) / threads_per_block;
    
    // Cap the number of blocks to avoid excessive overhead
    const int max_blocks = 65535;
    const int final_blocks = blocks > max_blocks ? max_blocks : blocks;
    
    fused_op_forward(final_blocks, threads_per_block,
                     A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
                     N, M);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op, "Broadcast multiplication fused operation");
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

def functional_model(
    A,
    B,
):
    # Move tensors to GPU if not already there
    if not A.is_cuda:
        A = A.cuda()
    if not B.is_cuda:
        B = B.cuda()
        
    # Create output tensor
    C = torch.empty(B.shape, device=B.device, dtype=B.dtype)
    
    # Call custom CUDA kernel
    fused_ext.fused_op(A, B, C)
    
    return C

M = 4096
N = 4096

def get_init_inputs():
    return []  # No special initialization inputs needed

def get_inputs():
    A = torch.rand(N).cuda()
    B = torch.rand(N, M).cuda()
    return [A, B]
