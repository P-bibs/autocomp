# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_202725/code_10.py
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# CUDA kernel source code
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void leaky_relu_vectorized_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    float negative_slope,
    int total_elements
) {
    // Grid-stride loop to handle all elements
    // Each thread processes 4 elements via vectorization
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    
    if (idx < total_elements) {
        // Load up to 4 floats at once (vectorized)
        int remaining = total_elements - idx;
        
        if (remaining >= 4) {
            // Vectorized load and process 4 elements
            float4 vals = *((const float4*)(input + idx));
            
            vals.x = (vals.x > 0.0f) ? vals.x : vals.x * negative_slope;
            vals.y = (vals.y > 0.0f) ? vals.y : vals.y * negative_slope;
            vals.z = (vals.z > 0.0f) ? vals.z : vals.z * negative_slope;
            vals.w = (vals.w > 0.0f) ? vals.w : vals.w * negative_slope;
            
            *((float4*)(output + idx)) = vals;
        } else {
            // Handle remaining elements (< 4)
            for (int i = 0; i < remaining; i++) {
                float val = input[idx + i];
                output[idx + i] = (val > 0.0f) ? val : val * negative_slope;
            }
        }
    }
}

void leaky_relu_vectorized_forward(
    torch::Tensor input,
    torch::Tensor output,
    float negative_slope
) {
    const int total_elements = input.numel();
    const int threads_per_block = 256;
    const int elements_per_thread = 4;
    
    // Calculate grid size: we process 4 elements per thread
    int blocks = (total_elements + threads_per_block * elements_per_thread - 1) / 
                 (threads_per_block * elements_per_thread);
    
    const float* input_ptr = input.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    
    leaky_relu_vectorized_kernel<<<blocks, threads_per_block>>>(
        input_ptr,
        output_ptr,
        negative_slope,
        total_elements
    );
    
    cudaDeviceSynchronize();
}
"""

# C++ binding code
cpp_source = r"""
#include <torch/extension.h>

void leaky_relu_vectorized_forward(
    torch::Tensor input,
    torch::Tensor output,
    float negative_slope
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("leaky_relu_forward", &leaky_relu_vectorized_forward, 
          "Vectorized Leaky ReLU forward pass");
}
"""

# Compile the extension
leaky_relu_ext = load_inline(
    name='leaky_relu_vectorized',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True,
    verbose=False
)

def functional_model(x, *, negative_slope):
    """
    Optimized leaky_relu using custom vectorized CUDA kernel.
    """
    output = torch.empty_like(x)
    leaky_relu_ext.leaky_relu_forward(x, output, negative_slope)
    return output

batch_size = 4096
dim = 393216

def get_init_inputs():
    return []

def get_inputs():
    x = torch.rand(batch_size, dim)
    return [x]
