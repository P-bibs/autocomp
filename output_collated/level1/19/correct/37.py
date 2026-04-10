# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_185153/code_14.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = []
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = []
REQUIRED_FLAT_STATE_NAMES = []


class ModelNew(nn.Module):
    """
    Simple model that performs a ReLU activation.
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

# CUDA Kernel for optimized ReLU
# This version uses a 1D grid with careful thread mapping to avoid alignment
# issues and reduce launch overhead while maintaining memory coalescing.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void relu_kernel(const float* __restrict__ input, float* __restrict__ output, size_t num_elements) {
    // Calculate global thread index
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    // Grid-stride loop to handle all elements efficiently
    for (size_t i = idx; i < num_elements; i += stride) {
        output[i] = fmaxf(0.0f, input[i]);
    }
}

void relu_launch(torch::Tensor input, torch::Tensor output) {
    const size_t num_elements = input.numel();
    
    // Use a more reasonable grid size to minimize launch overhead
    const int threads_per_block = 256;
    const int max_blocks = 65535; // Respect CUDA limits
    const int blocks = min(max_blocks, (int)((num_elements + threads_per_block - 1) / threads_per_block));

    relu_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        num_elements
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void relu_launch(torch::Tensor input, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("relu", &relu_launch, "Optimized ReLU kernel");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='optimized_relu',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x: torch.Tensor) -> torch.Tensor:
    """
    Optimized ReLU using a custom CUDA kernel that avoids alignment issues
    and reduces launch overhead compared to the original vectorized version.
    """
    output = torch.empty_like(x)
    fused_ext.relu(x, output)
    return output

# ----------------------------------------------------------------------
# Evaluation-setup constants (kept for completeness)
# ----------------------------------------------------------------------
batch_size = 4096
dim = 393216

def get_init_inputs():
    """Inputs required to initialise any global state (none here)."""
    return []

def get_inputs():
    """Generate a random input tensor on the GPU."""
    return [torch.rand(batch_size, dim, device='cuda')]
