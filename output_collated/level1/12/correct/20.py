# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_101900/code_17.py
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

# Optimization: Grid-Stride Loops
# The previous 2D grid approach was limited by block dimensions and potential 
# underutilization of GPU occupancy. By using a 1D grid-stride loop, we 
# decouple the kernel logic from the hardware's block size, allowing for 
# improved load balancing and latency hiding through software-pipelining.
# We also minimize integer division by using index mapping efficiently.

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_multiply_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ Out, long long total_elements, int M) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;

    // Grid-stride loop allows each thread to handle multiple elements
    // This improves cache reuse and masks memory access latency.
    for (long long i = idx; i < total_elements; i += stride) {
        int row = i / M;
        // B and Out share the same linear indexing
        Out[i] = A[row] * B[i];
    }
}

void fused_op_forward(torch::Tensor A, torch::Tensor B, torch::Tensor Out) {
    long long N = A.size(0);
    long long M = B.size(1);
    long long total_elements = N * M;
    
    // Using 256 threads per block is often the sweet spot for occupancy on 2080Ti.
    const int threadsPerBlock = 256;
    // Limit blocks to prevent over-subscription; 1024 blocks is generally sufficient for massive parallelism.
    const int maxBlocks = 1024;
    int blocksPerGrid = (total_elements + threadsPerBlock - 1) / threadsPerBlock;
    if (blocksPerGrid > maxBlocks) blocksPerGrid = maxBlocks;

    fused_multiply_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        A.data_ptr<float>(), 
        B.data_ptr<float>(), 
        Out.data_ptr<float>(), 
        total_elements, 
        (int)M
    );
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
    name='fused_mul_optimized',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(A, B):
    """
    Optimized functional model using a custom CUDA kernel with grid-stride loops.
    """
    N, M = B.shape
    # Pre-allocate output tensor on the same device as the inputs
    out = torch.empty((N, M), device=A.device, dtype=A.dtype)
    
    # Launch custom kernel via PyBind11 bindings
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
