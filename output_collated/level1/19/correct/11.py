# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_184018/code_7.py
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# CUDA kernel for vectorized ReLU using float4
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void relu_float4_kernel(const float* input, float* output, size_t num_elements) {
    // Calculate the number of float4 elements (each float4 contains 4 floats)
    size_t num_float4 = num_elements / 4;
    size_t remainder = num_elements % 4;
    
    // Grid-stride loop for float4 processing
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_float4; idx += blockDim.x * gridDim.x) {
        // Load 4 consecutive floats as a float4 vector
        float4 input_vec = reinterpret_cast<const float4*>(input)[idx];
        
        // Apply ReLU to each component
        float4 output_vec;
        output_vec.x = fmaxf(0.0f, input_vec.x);
        output_vec.y = fmaxf(0.0f, input_vec.y);
        output_vec.z = fmaxf(0.0f, input_vec.z);
        output_vec.w = fmaxf(0.0f, input_vec.w);
        
        // Store the result
        reinterpret_cast<float4*>(output)[idx] = output_vec;
    }
    
    // Handle remaining elements that don't fit in a float4
    if (threadIdx.x == 0 && blockIdx.x == 0 && remainder > 0) {
        for (size_t i = num_float4 * 4; i < num_elements; ++i) {
            output[i] = fmaxf(0.0f, input[i]);
        }
    }
}

void launch_relu_float4(const float* input, float* output, size_t num_elements) {
    // Calculate optimal grid and block dimensions
    const int block_size = 256;
    const int max_blocks = 65535;
    
    // Calculate number of blocks needed for float4 elements
    size_t num_float4 = num_elements / 4;
    int num_blocks = (num_float4 + block_size - 1) / block_size;
    num_blocks = min(num_blocks, max_blocks);
    
    // Ensure at least one block for handling remainder
    num_blocks = max(num_blocks, 1);
    
    // Launch the kernel
    relu_float4_kernel<<<num_blocks, block_size>>>(input, output, num_elements);
}
"""

# C++ binding code
cpp_source = r"""
#include <torch/extension.h>

void launch_relu_float4(const float* input, float* output, size_t num_elements);

void fused_relu_forward(torch::Tensor input, torch::Tensor output) {
    const float* input_ptr = input.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    size_t num_elements = input.numel();
    
    launch_relu_float4(input_ptr, output_ptr, num_elements);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_relu", &fused_relu_forward, "Fused ReLU with float4 vectorization");
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
    # Create output tensor with the same shape and device as input
    output = torch.empty_like(x)
    
    # Call the custom CUDA kernel
    fused_ext.fused_relu(x, output)
    
    return output

batch_size = 4096
dim = 393216

def get_init_inputs():
    return []  # No special initialization inputs needed

def get_inputs():
    x = torch.rand(batch_size, dim)
    return [x]
