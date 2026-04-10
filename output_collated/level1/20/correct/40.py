# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_204858/code_12.py
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

# -------------------------------------------------------------------------
# CUDA kernel – optimized with grid-stride loop for better GPU utilization
# -------------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void leaky_relu_vec4_kernel(const float* __restrict__ input,
                                        float* __restrict__ output,
                                        float negative_slope,
                                        int numel) {
    // Grid-stride loop: each thread processes multiple elements
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    
    // Process multiple vec4 elements per thread
    while (idx + 3 < numel) {
        // Load 4 elements at once
        float4 in_vec = reinterpret_cast<const float4*>(input)[idx / 4];
        float4 out_vec;

        // Apply Leaky ReLU
        out_vec.x = in_vec.x > 0.0f ? in_vec.x : in_vec.x * negative_slope;
        out_vec.y = in_vec.y > 0.0f ? in_vec.y : in_vec.y * negative_slope;
        out_vec.z = in_vec.z > 0.0f ? in_vec.z : in_vec.z * negative_slope;
        out_vec.w = in_vec.w > 0.0f ? in_vec.w : in_vec.w * negative_slope;

        // Store result
        reinterpret_cast<float4*>(output)[idx / 4] = out_vec;
        
        // Move to next set of elements (grid-stride)
        idx += blockDim.x * gridDim.x * 4;
    }
    
    // Handle remaining elements that don't form a complete float4
    for (int i = idx; i < numel; ++i) {
        float val = input[i];
        output[i] = val > 0.0f ? val : val * negative_slope;
    }
}

void launch_leaky_relu(int numel, float* input, float* output, float negative_slope) {
    const int block_size = 1024;
    
    // Calculate grid size to ensure we have enough blocks
    int vec_elems = (numel + 3) / 4;  // ceil(numel/4)
    int grid_size = (vec_elems + block_size - 1) / block_size;
    
    // Cap grid size to avoid excessive kernel launch overhead
    const int max_grid = 65535;
    if (grid_size > max_grid) grid_size = max_grid;
    
    leaky_relu_vec4_kernel<<<grid_size, block_size>>>(input, output, negative_slope, numel);
}
"""

# -------------------------------------------------------------------------
# C++ binding – exposes the kernel to Python
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void launch_leaky_relu(int numel, float* input, float* output, float negative_slope);

void leaky_relu_forward(torch::Tensor input, torch::Tensor output, float negative_slope) {
    launch_leaky_relu(input.numel(),
                      input.data_ptr<float>(),
                      output.data_ptr<float>(),
                      negative_slope);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("leaky_relu_forward", &leaky_relu_forward, "Vectorized Leaky ReLU forward with grid-stride loop");
}
"""

# -------------------------------------------------------------------------
# Compile the inline extension
# -------------------------------------------------------------------------
leaky_relu_ext = load_inline(
    name='leaky_relu_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# -------------------------------------------------------------------------
# Model parameters
# -------------------------------------------------------------------------
batch_size = 4096
dim = 393216

def get_init_inputs():
    return []

def get_inputs():
    # Create a random input tensor on the GPU
    x = torch.rand(batch_size, dim, device='cuda', dtype=torch.float32)
    return [x]

# -------------------------------------------------------------------------
# Functional model – uses grid-stride loop for optimal GPU utilization
# -------------------------------------------------------------------------
def functional_model(x, *, negative_slope):
    """
    In-place leaky ReLU with grid-stride loop optimization.
    
    The grid-stride loop allows the GPU to process all elements efficiently
    by having each thread process multiple elements, ensuring better utilization
    even when the grid size is limited.
    """
    # Ensure GPU memory is contiguous for efficient coalesced access
    if not x.is_contiguous():
        x = x.contiguous()

    # In-place call: input and output point to the same memory
    leaky_relu_ext.leaky_relu_forward(x, x, float(negative_slope))
    return x
