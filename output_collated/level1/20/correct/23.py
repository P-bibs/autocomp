# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_202725/code_27.py
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

# CUDA Kernel for Vectorized Leaky ReLU
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector_types.h>

__global__ void leaky_relu_vectorized_kernel(const float* __restrict__ input, 
                                             float* __restrict__ output, 
                                             float negative_slope, 
                                             size_t n) {
    // Vectorized index calculation: each thread handles 4 elements
    size_t vec_n = n / 4;
    size_t idx = (blockIdx.x * blockDim.x + threadIdx.x);
    
    if (idx < vec_n) {
        // Load 128 bits (4 x 32-bit floats) from global memory
        float4* in_vec = (float4*)input;
        float4* out_vec = (float4*)output;
        float4 vals = in_vec[idx];
        
        // Unrolled Leaky ReLU operations
        vals.x = (vals.x > 0.0f) ? vals.x : vals.x * negative_slope;
        vals.y = (vals.y > 0.0f) ? vals.y : vals.y * negative_slope;
        vals.z = (vals.z > 0.0f) ? vals.z : vals.z * negative_slope;
        vals.w = (vals.w > 0.0f) ? vals.w : vals.w * negative_slope;
        
        out_vec[idx] = vals;
    }
    
    // Handle tail elements (remaining elements if n is not a multiple of 4)
    size_t remaining_start = vec_n * 4;
    for (size_t i = remaining_start + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        float val = input[i];
        output[i] = (val > 0.0f) ? val : val * negative_slope;
    }
}

void leaky_relu_forward(torch::Tensor input, torch::Tensor output, float negative_slope) {
    const size_t n = input.numel();
    const int threads = 256;
    const int blocks = (n / 4 + threads - 1) / threads;
    
    leaky_relu_vectorized_kernel<<<blocks, threads>>>(
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
    m.def("leaky_relu", &leaky_relu_forward, "Vectorized Leaky ReLU forward kernel");
}
"""

# Compile the extension inline
leaky_relu_ext = load_inline(
    name='leaky_relu_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math', '-arch=sm_75'],
    with_cuda=True
)

def functional_model(x, *, negative_slope):
    """
    Optimized functional_model using vectorized CUDA kernel.
    Ensures input is contiguous and float32 for 128-bit memory alignment.
    """
    if x.dtype != torch.float32:
        x = x.to(torch.float32)
    # Ensure memory is contiguous to allow valid float4 pointer casting
    if not x.is_contiguous():
        x = x.contiguous()
    
    output = torch.empty_like(x)
    leaky_relu_ext.leaky_relu(x, output, float(negative_slope))
    return output
