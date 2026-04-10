# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_204858/code_10.py
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

# CUDA Kernel with float4 vectorization and loop unrolling
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void leaky_relu_vectorized_kernel(const float* __restrict__ input, float* __restrict__ output, float negative_slope, size_t n) {
    // Process 16 floats per thread (4 float4 vectors) to improve ILP and reduce kernel launch overhead
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t idx = tid * 16;
    
    if (idx + 15 < n) {
        // All 16 elements are within bounds - process 4 float4 vectors
        float4 in_vec0 = reinterpret_cast<const float4*>(input)[idx / 4];
        float4 in_vec1 = reinterpret_cast<const float4*>(input)[idx / 4 + 1];
        float4 in_vec2 = reinterpret_cast<const float4*>(input)[idx / 4 + 2];
        float4 in_vec3 = reinterpret_cast<const float4*>(input)[idx / 4 + 3];
        
        float4 out_vec0, out_vec1, out_vec2, out_vec3;

        // Apply Leaky ReLU with manual loop unrolling for better ILP
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            float val = (&in_vec0.x)[i];
            (&out_vec0.x)[i] = (val > 0.0f) ? val : val * negative_slope;
        }
        
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            float val = (&in_vec1.x)[i];
            (&out_vec1.x)[i] = (val > 0.0f) ? val : val * negative_slope;
        }
        
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            float val = (&in_vec2.x)[i];
            (&out_vec2.x)[i] = (val > 0.0f) ? val : val * negative_slope;
        }
        
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            float val = (&in_vec3.x)[i];
            (&out_vec3.x)[i] = (val > 0.0f) ? val : val * negative_slope;
        }

        reinterpret_cast<float4*>(output)[idx / 4] = out_vec0;
        reinterpret_cast<float4*>(output)[idx / 4 + 1] = out_vec1;
        reinterpret_cast<float4*>(output)[idx / 4 + 2] = out_vec2;
        reinterpret_cast<float4*>(output)[idx / 4 + 3] = out_vec3;
    } else {
        // Handle remaining elements individually to avoid out-of-bounds access
        for (int i = 0; i < 16 && idx + i < n; ++i) {
            float val = input[idx + i];
            output[idx + i] = (val > 0.0f) ? val : val * negative_slope;
        }
    }
}

void leaky_relu_forward(torch::Tensor input, torch::Tensor output, float negative_slope) {
    const size_t n = input.numel();
    // Use 1/16 the threads since each processes 16 elements
    const int threads = 256;
    const int blocks = (n / 16 + threads - 1) / threads;
    
    leaky_relu_vectorized_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        negative_slope,
        n
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void leaky_relu_forward(torch::Tensor input, torch::Tensor output, float negative_slope);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("leaky_relu", &leaky_relu_forward, "Vectorized and Unrolled Leaky ReLU forward");
}
"""

leaky_relu_ext = load_inline(
    name='leaky_relu_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, negative_slope):
    """
    Optimized functional_model using vectorized and unrolled CUDA kernel.
    """
    # Ensure input is contiguous and float32
    if not x.is_contiguous():
        x = x.contiguous()
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
