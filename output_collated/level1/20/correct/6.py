# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_202725/code_3.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['negative_slope']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['negative_slope']
REQUIRED_FLAT_STATE_NAMES = []


class ModelNew(nn.Module):
    """
    Simple model that performs a LeakyReLU activation.
    """

    def __init__(self, negative_slope: float=0.01):
        """
        Initializes the LeakyReLU module.

        Args:
            negative_slope (float, optional): The negative slope of the activation function. Defaults to 0.01.
        """
        super(ModelNew, self).__init__()
        self.negative_slope = negative_slope

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
    if 'negative_slope' in flat_state:
        state_kwargs['negative_slope'] = flat_state['negative_slope']
    else:
        state_kwargs['negative_slope'] = getattr(model, 'negative_slope')
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

# CUDA Kernel for Leaky ReLU with vectorized memory access and loop unrolling
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void leaky_relu_kernel(const float* input, float* output, float negative_slope, size_t n) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    // Process multiple elements per thread with loop unrolling
    for (size_t idx = tid * 4; idx < n && idx + 3 < n; idx += stride * 4) {
        // Vectorized load of 4 floats
        float4 input_vec = reinterpret_cast<const float4*>(&input[idx])[0];
        float4 output_vec;
        
        // Apply Leaky ReLU with conditional expressions
        output_vec.x = (input_vec.x > 0.0f) ? input_vec.x : input_vec.x * negative_slope;
        output_vec.y = (input_vec.y > 0.0f) ? input_vec.y : input_vec.y * negative_slope;
        output_vec.z = (input_vec.z > 0.0f) ? input_vec.z : input_vec.z * negative_slope;
        output_vec.w = (input_vec.w > 0.0f) ? input_vec.w : input_vec.w * negative_slope;
        
        // Vectorized store of 4 floats
        reinterpret_cast<float4*>(&output[idx])[0] = output_vec;
    }
    
    // Handle remaining elements
    size_t start_idx = (n / 4) * 4;
    for (size_t idx = start_idx + tid; idx < n; idx += stride) {
        float val = input[idx];
        output[idx] = (val > 0.0f) ? val : val * negative_slope;
    }
}

void leaky_relu_forward(torch::Tensor input, torch::Tensor output, float negative_slope) {
    const size_t n = input.numel();
    
    // Optimize for the RTX 2080Ti with 46 streaming multiprocessors
    const int threads_per_block = 256;
    const int blocks_per_sm = 4;  // Empirically determined for good occupancy
    const int blocks = min(static_cast<int>((n + 1023) / 1024), 
                          blocks_per_sm * 46); // 46 SMs on RTX 2080Ti
    
    leaky_relu_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        negative_slope,
        n
    );
}
"""

# C++ Interface
cpp_source = r"""
#include <torch/extension.h>

void leaky_relu_forward(torch::Tensor input, torch::Tensor output, float negative_slope);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("leaky_relu", &leaky_relu_forward, "Leaky ReLU forward kernel");
}
"""

# Compile the extension
leaky_relu_ext = load_inline(
    name='leaky_relu_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, negative_slope):
    """
    Optimized functional_model using custom CUDA kernel with vectorized memory access
    and loop unrolling for maximum memory bandwidth utilization.
    """
    # Ensure input is float32 to match kernel expectations
    if x.dtype != torch.float32:
        x = x.to(torch.float32)
    
    output = torch.empty_like(x)
    leaky_relu_ext.leaky_relu(x, output, float(negative_slope))
    return output

# --- Constants and helper functions as required ---
batch_size = 4096
dim = 393216

def get_init_inputs():
    return []

def get_inputs():
    # Use float32 to ensure compatibility with our custom CUDA kernel
    x = torch.rand(batch_size, dim, device='cuda', dtype=torch.float32)
    return [x]
