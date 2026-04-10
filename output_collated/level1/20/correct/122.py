# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_222822/code_9.py
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

# ----------------------------------------------------------------------
# CUDA kernel – Optimized with loop unrolling (16 floats per thread)
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void leaky_relu_vectorized_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    float negative_slope,
    size_t n)
{
    // Each thread now processes 16 consecutive floats using 4 float4s
    size_t base_idx = (size_t)(blockIdx.x * blockDim.x + threadIdx.x) * 16;

    if (base_idx + 15 < n) {
        // Vectorized loads from global memory (4 x float4 = 16 floats)
        float4 in_vec0 = reinterpret_cast<const float4*>(input)[base_idx / 4 + 0];
        float4 in_vec1 = reinterpret_cast<const float4*>(input)[base_idx / 4 + 1];
        float4 in_vec2 = reinterpret_cast<const float4*>(input)[base_idx / 4 + 2];
        float4 in_vec3 = reinterpret_cast<const float4*>(input)[base_idx / 4 + 3];

        // Apply Leaky ReLU logic to each element in the 4 float4s
        float4 out_vec0, out_vec1, out_vec2, out_vec3;

        out_vec0.x = (in_vec0.x > 0.0f) ? in_vec0.x : in_vec0.x * negative_slope;
        out_vec0.y = (in_vec0.y > 0.0f) ? in_vec0.y : in_vec0.y * negative_slope;
        out_vec0.z = (in_vec0.z > 0.0f) ? in_vec0.z : in_vec0.z * negative_slope;
        out_vec0.w = (in_vec0.w > 0.0f) ? in_vec0.w : in_vec0.w * negative_slope;

        out_vec1.x = (in_vec1.x > 0.0f) ? in_vec1.x : in_vec1.x * negative_slope;
        out_vec1.y = (in_vec1.y > 0.0f) ? in_vec1.y : in_vec1.y * negative_slope;
        out_vec1.z = (in_vec1.z > 0.0f) ? in_vec1.z : in_vec1.z * negative_slope;
        out_vec1.w = (in_vec1.w > 0.0f) ? in_vec1.w : in_vec1.w * negative_slope;

        out_vec2.x = (in_vec2.x > 0.0f) ? in_vec2.x : in_vec2.x * negative_slope;
        out_vec2.y = (in_vec2.y > 0.0f) ? in_vec2.y : in_vec2.y * negative_slope;
        out_vec2.z = (in_vec2.z > 0.0f) ? in_vec2.z : in_vec2.z * negative_slope;
        out_vec2.w = (in_vec2.w > 0.0f) ? in_vec2.w : in_vec2.w * negative_slope;

        out_vec3.x = (in_vec3.x > 0.0f) ? in_vec3.x : in_vec3.x * negative_slope;
        out_vec3.y = (in_vec3.y > 0.0f) ? in_vec3.y : in_vec3.y * negative_slope;
        out_vec3.z = (in_vec3.z > 0.0f) ? in_vec3.z : in_vec3.z * negative_slope;
        out_vec3.w = (in_vec3.w > 0.0f) ? in_vec3.w : in_vec3.w * negative_slope;

        // Vectorized stores to global memory
        reinterpret_cast<float4*>(output)[base_idx / 4 + 0] = out_vec0;
        reinterpret_cast<float4*>(output)[base_idx / 4 + 1] = out_vec1;
        reinterpret_cast<float4*>(output)[base_idx / 4 + 2] = out_vec2;
        reinterpret_cast<float4*>(output)[base_idx / 4 + 3] = out_vec3;
    } else {
        // Tail handling loop for remaining elements (when n is not a multiple of 16)
        for (int i = 0; i < 16; ++i) {
            size_t current_idx = base_idx + i;
            if (current_idx < n) {
                float val = input[current_idx];
                output[current_idx] = (val > 0.0f) ? val : val * negative_slope;
            }
        }
    }
}

void leaky_relu_forward(torch::Tensor input, torch::Tensor output, float negative_slope) {
    const size_t n = input.numel();
    
    // Optimization: Use 1024 threads per block to maximize occupancy
    const int block_size = 1024;
    // Grid size adjusted for 16 elements per thread
    const int grid_size = (n + (block_size * 16) - 1) / (block_size * 16);
    
    leaky_relu_vectorized_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        negative_slope,
        n
    );
}
"""

# ----------------------------------------------------------------------
# C++ binding
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>
void leaky_relu_forward(torch::Tensor input, torch::Tensor output, float negative_slope);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("leaky_relu", &leaky_relu_forward, "Vectorized Leaky ReLU forward with loop unrolling");
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
    Optimized functional_model using the compiled CUDA kernel with loop unrolling.
    Each thread now processes 16 floats instead of 4.
    """
    # Ensure input is contiguous for aligned memory access
    if not x.is_contiguous():
        x = x.contiguous()
    
    # Cast to float32 as required by the kernel implementation
    if x.dtype != torch.float32:
        x = x.to(torch.float32)
    
    output = torch.empty_like(x)
    leaky_relu_ext.leaky_relu(x, output, float(negative_slope))
    return output

# Constants for evaluation environment
batch_size = 4096
dim = 393216

def get_init_inputs():
    return []

def get_inputs():
    x = torch.rand(batch_size, dim, device='cuda', dtype=torch.float32)
    return [x]
