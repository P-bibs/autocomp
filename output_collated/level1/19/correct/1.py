# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_183549/code_2.py
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
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the CUDA kernel for ReLU operation
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void relu_forward_kernel(const float* input, float* output, int64_t numel) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = blockDim.x * gridDim.x;
    
    for (int64_t i = idx; i < numel; i += stride) {
        output[i] = fmaxf(input[i], 0.0f);
    }
}

void relu_forward(int blocks, int threads, const torch::Tensor& input, torch::Tensor& output) {
    relu_forward_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        input.numel()
    );
}
"""

# Define the C++ interface
cpp_source = r"""
#include <torch/extension.h>

void relu_forward(int blocks, int threads, const torch::Tensor& input, torch::Tensor& output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("relu_forward", &relu_forward, "ReLU forward pass with custom CUDA kernel");
}
"""

# Compile the extension with optimization flags
fused_ext = load_inline(
    name='fused_relu',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x):
    # Create output tensor with the same shape as input
    output = torch.empty_like(x)
    
    # Calculate grid and block dimensions
    numel = x.numel()
    threads_per_block = 1024
    blocks = min(65535, (numel + threads_per_block - 1) // threads_per_block)
    
    # Ensure tensors are on CUDA
    if not x.is_cuda:
        x = x.cuda()
    if not output.is_cuda:
        output = output.cuda()
    
    # Call the custom CUDA kernel
    fused_ext.relu_forward(blocks, threads_per_block, x, output)
    
    return output

batch_size = 4096
dim = 393216

def get_init_inputs():
    return []  # No special initialization inputs needed

def get_inputs():
    x = torch.rand(batch_size, dim)
    return [x]
