# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_105940/code_3.py
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

# Custom CUDA kernel that fuses unsqueeze and multiply operations
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_unsqueeze_mul_kernel(
    const float* A,
    const float* B,
    float* output,
    int N,
    int M
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * M;
    
    if (idx < total_elements) {
        int n = idx / M;  // row index
        int m = idx % M;  // column index
        
        // Directly compute A[n] * B[n * M + m] without intermediate tensor
        output[idx] = A[n] * B[idx];
    }
}

void fused_unsqueeze_mul_forward(int blocks, int threads, 
                                const float* A, const float* B, float* output,
                                int N, int M) {
    fused_unsqueeze_mul_kernel<<<blocks, threads>>>(A, B, output, N, M);
}
"""

# C++ binding
cpp_source = r"""
#include <torch/extension.h>

void fused_unsqueeze_mul_forward(int blocks, int threads,
                                const float* A, const float* B, float* output,
                                int N, int M);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_unsqueeze_mul", &fused_unsqueeze_mul_forward, "Fused unsqueeze and multiply operation");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_unsqueeze_mul',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(A, B):
    N, M = B.shape
    
    # Ensure inputs are on GPU and contiguous
    A_gpu = A.cuda().contiguous()
    B_gpu = B.cuda().contiguous()
    
    # Allocate output tensor
    output = torch.empty(N, M, dtype=torch.float32, device='cuda')
    
    # Calculate grid and block dimensions
    total_elements = N * M
    threads_per_block = 256
    blocks = (total_elements + threads_per_block - 1) // threads_per_block
    
    # Launch kernel
    fused_ext.fused_unsqueeze_mul(blocks, threads_per_block, 
                                  A_gpu.data_ptr<float>(), 
                                  B_gpu.data_ptr<float>(), 
                                  output.data_ptr<float>(),
                                  N, M)
    
    return output

M = 4096
N = 4096

def get_init_inputs():
    return []  # No special initialization inputs needed

def get_inputs():
    A = torch.rand(N)
    B = torch.rand(N, M)
    return [A, B]
