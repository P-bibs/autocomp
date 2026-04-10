# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_184018/code_2.py
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

# CUDA kernel with optimized grid-stride loop for RTX 2080Ti
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ __launch_bounds__(1024) void relu_kernel(const float* __restrict__ input, float* __restrict__ output, int64_t num_elements) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    
    for (int64_t i = idx; i < num_elements; i += stride) {
        float val = input[i];
        output[i] = fmaxf(0.0f, val);
    }
}

void relu_forward_cuda(const torch::Tensor& input, torch::Tensor& output) {
    const int64_t num_elements = input.numel();
    const int threads = 1024;  // Optimal for RTX 2080Ti (Turing architecture)
    const int blocks = (num_elements + threads - 1) / threads;
    
    relu_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        num_elements
    );
}
"""

# C++ interface
cpp_source = r"""
#include <torch/extension.h>

void relu_forward_cuda(const torch::Tensor& input, torch::Tensor& output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_relu", &relu_forward_cuda, "Optimized C++ ReLU Forward");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_relu_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x):
    # Ensure input is on device
    if not x.is_cuda:
        x = x.to('cuda', non_blocking=True)
    
    # Pre-allocate output to avoid PyTorch framework overhead inside the kernel launcher
    output = torch.empty_like(x)
    
    # Call the optimized C++/CUDA kernel
    fused_ext.fused_relu(x, output)
    
    return output

batch_size = 4096
dim = 393216

def get_init_inputs():
    return []

def get_inputs():
    # Return inputs on CPU to match original behavior, 
    # the function converts them internally
    x = torch.rand(batch_size, dim)
    return [x]
