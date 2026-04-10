# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_231959/code_17.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = []
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = []
REQUIRED_FLAT_STATE_NAMES = []


class ModelNew(nn.Module):
    """
    Simple model that performs a Tanh activation.
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
# We use a grid-stride loop to handle arbitrary sizes while maintaining
# memory coalescing and minimizing kernel launch overhead.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void tanh_kernel(const float* __restrict__ input, float* __restrict__ output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    for (; idx < n; idx += stride) {
        // Use __tanhf for hardware-accelerated approximation via fast-math,
        // or standard tanhf for full precision. __use_fast_math flag is provided.
        output[idx] = tanhf(input[idx]);
    }
}

void launch_tanh(torch::Tensor input, torch::Tensor output) {
    const size_t n = input.numel();
    const int threads = 256;
    // Cap the blocks to prevent excessive scheduling overhead, 
    // grid-stride loop handles the rest.
    const int blocks = std::min((size_t)1024, (n + threads - 1) / threads);
    
    tanh_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        n
    );
}
"""

# --- C++ Binding ---
# Using pybind11 to expose the kernel to Python
cpp_source = r"""
#include <torch/extension.h>

void launch_tanh(torch::Tensor input, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_tanh", &launch_tanh, "Custom Tanh Kernel");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='tanh_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x):
    # Ensure contiguous layout to allow coalesced memory access
    if not x.is_contiguous():
        x = x.contiguous()
    
    output = torch.empty_like(x)
    fused_ext.launch_tanh(x, output)
    return output

# --- Setup for evaluation environment ---
batch_size = 4096
dim = 393216

def get_init_inputs():
    return []

def get_inputs():
    # Ensure inputs are allocated on GPU as required by the custom kernel
    x = torch.rand(batch_size, dim, device='cuda', dtype=torch.float32)
    return [x]
