# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_105940/code_19.py
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
# Optimization: The kernel maps the 1D thread ID to 2D coordinates.
# Since A[n] relies on 'n' (block) and B[idx] is contiguous in memory, 
# this kernel achieves 100% coalesced memory access for B and output.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_unsqueeze_mul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ output,
    int N,
    int M
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < (long long)N * M) {
        // n = idx / M is optimized by the compiler using reciprocal multiplication
        // idx % M is the offset within the row
        int n = idx / M;
        output[idx] = A[n] * B[idx];
    }
}

void fused_unsqueeze_mul_forward(
    const torch::Tensor& A, 
    const torch::Tensor& B, 
    torch::Tensor& output
) {
    int N = A.size(0);
    int M = B.size(1);
    long long total_elements = (long long)N * M;
    
    int threads_per_block = 256;
    int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    fused_unsqueeze_mul_kernel<<<blocks, threads_per_block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        output.data_ptr<float>(),
        N, M
    );
}
"""

# --- C++ Binding ---
cpp_source = r"""
#include <torch/extension.h>

void fused_unsqueeze_mul_forward(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& output);

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
    """
    Optimized functional_model that uses a custom CUDA kernel to fuse
    unsqueeze and broadcast multiplication.
    """
    # Ensure inputs are on GPU and contiguous to allow direct pointer access
    if not A.is_cuda: A = A.cuda()
    if not B.is_cuda: B = B.cuda()
    
    A = A.contiguous()
    B = B.contiguous()
    
    N, M = B.shape
    
    # Pre-allocate output
    output = torch.empty((N, M), dtype=torch.float32, device=B.device)
    
    # Launch fused kernel
    fused_ext.fused_unsqueeze_mul(A, B, output)
    
    return output

# --- Setup Constants ---
M = 4096
N = 4096

def get_init_inputs():
    return []

def get_inputs():
    A = torch.rand(N, device='cuda')
    B = torch.rand(N, M, device='cuda')
    return [A, B]
