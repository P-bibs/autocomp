# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_103853/code_3.py
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

# Custom CUDA kernel for fused broadcast multiplication
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_broadcast_mul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ output,
    int N,
    int M
) {
    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Each thread processes multiple elements for better memory coalescing
    for (int i = idx; i < N * M; i += stride) {
        int n = i / M;  // Row index
        int m = i % M;  // Column index
        
        // Coalesced access: consecutive threads access consecutive memory locations in B and output
        output[i] = A[n] * B[i];
    }
}

void fused_broadcast_mul_forward(int blocks, int threads, 
                                const float* A, const float* B, float* output,
                                int N, int M) {
    fused_broadcast_mul_kernel<<<blocks, threads>>>(A, B, output, N, M);
}
"""

# C++ binding code
cpp_source = r"""
#include <torch/extension.h>

void fused_broadcast_mul_forward(int blocks, int threads,
                                const float* A, const float* B, float* output,
                                int N, int M);

void fused_broadcast_mul(torch::Tensor A, torch::Tensor B, torch::Tensor output) {
    const int N = A.size(0);
    const int M = B.size(1);
    
    // Launch configuration optimized for memory coalescing
    const int threads_per_block = 256;  // Multiple of 32 (warp size)
    const int blocks = (N * M + threads_per_block - 1) / threads_per_block;
    
    fused_broadcast_mul_forward(blocks, threads_per_block,
                               A.data_ptr<float>(), B.data_ptr<float>(), output.data_ptr<float>(),
                               N, M);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_broadcast_mul", &fused_broadcast_mul, "Fused broadcast multiplication");
}
"""

# Compile the extension with optimization flags
fused_ext = load_inline(
    name='fused_broadcast_mul_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(A, B):
    # Ensure inputs are on GPU and have correct dtype
    if not A.is_cuda:
        A = A.cuda()
    if not B.is_cuda:
        B = B.cuda()
    
    A = A.to(torch.float32)
    B = B.to(torch.float32)
    
    # Create output tensor
    output = torch.empty(B.shape, dtype=torch.float32, device='cuda')
    
    # Call custom CUDA kernel
    fused_ext.fused_broadcast_mul(A, B, output)
    
    return output

M = 4096
N = 4096

def get_init_inputs():
    return []  # No special initialization inputs needed

def get_inputs():
    A = torch.rand(N)
    B = torch.rand(N, M)
    return [A, B]
