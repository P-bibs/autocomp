# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_213956/code_9.py
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

# Optimized CUDA Kernel for Vectorized Leaky ReLU with increased vectorization and optimized block size
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector_types.h>

__global__ void leaky_relu_vectorized_kernel(const float* input, float* output, 
                                              float negative_slope, size_t n) {
    // Each thread now processes 8 elements using two float4 vectorizations
    size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    
    if (idx + 8 <= n) {
        // Load first 4 floats as a single float4 (128-bit memory transaction)
        float4 vals1 = *reinterpret_cast<const float4*>(&input[idx]);
        // Load next 4 floats as another float4
        float4 vals2 = *reinterpret_cast<const float4*>(&input[idx + 4]);
        
        // Apply leaky ReLU directly to output to reduce register pressure
        float4 result1, result2;
        
        // Process first float4
        result1.x = (vals1.x > 0.0f) ? vals1.x : __fmul_rn(vals1.x, negative_slope);
        result1.y = (vals1.y > 0.0f) ? vals1.y : __fmul_rn(vals1.y, negative_slope);
        result1.z = (vals1.z > 0.0f) ? vals1.z : __fmul_rn(vals1.z, negative_slope);
        result1.w = (vals1.w > 0.0f) ? vals1.w : __fmul_rn(vals1.w, negative_slope);
        
        // Process second float4
        result2.x = (vals2.x > 0.0f) ? vals2.x : __fmul_rn(vals2.x, negative_slope);
        result2.y = (vals2.y > 0.0f) ? vals2.y : __fmul_rn(vals2.y, negative_slope);
        result2.z = (vals2.z > 0.0f) ? vals2.z : __fmul_rn(vals2.z, negative_slope);
        result2.w = (vals2.w > 0.0f) ? vals2.w : __fmul_rn(vals2.w, negative_slope);
        
        // Store results
        *reinterpret_cast<float4*>(&output[idx]) = result1;
        *reinterpret_cast<float4*>(&output[idx + 4]) = result2;
    }
    else {
        // Completely unrolled handling of remaining elements (less than 8)
        // This eliminates loop overhead and divergence
        size_t element_idx = idx;
        if (element_idx < n) {
            float val = input[element_idx];
            output[element_idx] = (val > 0.0f) ? val : __fmul_rn(val, negative_slope);
        }
        element_idx++;
        if (element_idx < n) {
            float val = input[element_idx];
            output[element_idx] = (val > 0.0f) ? val : __fmul_rn(val, negative_slope);
        }
        element_idx++;
        if (element_idx < n) {
            float val = input[element_idx];
            output[element_idx] = (val > 0.0f) ? val : __fmul_rn(val, negative_slope);
        }
        element_idx++;
        if (element_idx < n) {
            float val = input[element_idx];
            output[element_idx] = (val > 0.0f) ? val : __fmul_rn(val, negative_slope);
        }
        element_idx++;
        if (element_idx < n) {
            float val = input[element_idx];
            output[element_idx] = (val > 0.0f) ? val : __fmul_rn(val, negative_slope);
        }
        element_idx++;
        if (element_idx < n) {
            float val = input[element_idx];
            output[element_idx] = (val > 0.0f) ? val : __fmul_rn(val, negative_slope);
        }
        element_idx++;
        if (element_idx < n) {
            float val = input[element_idx];
            output[element_idx] = (val > 0.0f) ? val : __fmul_rn(val, negative_slope);
        }
        element_idx++;
        if (element_idx < n) {
            float val = input[element_idx];
            output[element_idx] = (val > 0.0f) ? val : __fmul_rn(val, negative_slope);
        }
    }
}

void leaky_relu_forward(torch::Tensor input, torch::Tensor output, float negative_slope) {
    const size_t n = input.numel();
    const int threads = 256;  // Optimized for RTX 2080Ti - better occupancy for memory-bound kernels
    const int blocks = (n / 8 + threads - 1) / threads;  // Each thread handles 8 elements now
    
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
    m.def("leaky_relu", &leaky_relu_forward, "Optimized Vectorized Leaky ReLU forward kernel with improved occupancy");
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
    Optimized functional_model using vectorized CUDA kernel with improved occupancy.
    Processes 8 elements per thread using two float4 loads/stores for better memory throughput.
    """
    # Ensure input is float32 and contiguous for optimal memory access
    if x.dtype != torch.float32:
        x = x.to(torch.float32)
    if not x.is_contiguous():
        x = x.contiguous()
    
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
