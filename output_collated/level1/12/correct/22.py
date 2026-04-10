# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_101900/code_19.py
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

# Optimized CUDA kernel using Grid-Stride Loops
# 1. Thread-level load imbalance is handled by the loop.
# 2. Reduced branching compared to the original if-guard approach.
# 3. Each thread processes multiple elements, maximizing throughput.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void broadcast_mul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ output,
    int total_elements,
    int M
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Grid-stride loop: each thread processes a chunk of the data
    for (int i = idx; i < total_elements; i += stride) {
        int n = i / M;
        output[i] = A[n] * B[i];
    }
}

void broadcast_mul_forward(
    const torch::Tensor& A,
    const torch::Tensor& B,
    torch::Tensor& output
) {
    const int total_elements = output.numel();
    const int M = B.size(1);
    
    // Launch configuration:
    // Using a fixed number of threads and a calculated number of blocks.
    // The grid-stride loop makes the code resilient to varying block grid sizes.
    const int threads_per_block = 256;
    const int blocks = std::min((total_elements + threads_per_block - 1) / threads_per_block, 1024);
    
    broadcast_mul_kernel<<<blocks, threads_per_block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        output.data_ptr<float>(),
        total_elements,
        M
    );
}
"""

# C++ interface bindings
cpp_source = r"""
#include <torch/extension.h>

void broadcast_mul_forward(
    const torch::Tensor& A,
    const torch::Tensor& B,
    torch::Tensor& output
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("broadcast_mul", &broadcast_mul_forward, "Grid-stride broadcast multiplication kernel");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='broadcast_mul_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(A, B):
    """
    Optimized functional model performing A * B (broadcasted)
    using a custom grid-stride CUDA kernel.
    """
    # Ensure inputs are contiguous and on GPU
    A = A.contiguous().cuda()
    B = B.contiguous().cuda()
    
    # Create output tensor on the same device
    output = torch.empty_like(B)
    
    # Call the optimized custom CUDA kernel
    fused_ext.broadcast_mul(A, B, output)
    
    return output

# Parameters for benchmarking/execution
M = 4096
N = 4096

def get_init_inputs():
    return []

def get_inputs():
    A = torch.rand(N)
    B = torch.rand(N, M)
    return [A, B]
