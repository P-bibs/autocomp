# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_202725/code_28.py
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
# CUDA kernel: In-place Vectorized Leaky ReLU
# Improvements:
# 1. In-place modification to avoid massive device memory allocation.
# 2. Block size set to 1024 for higher occupancy.
# 3. Vectorized float4 memory access for maximum DRAM bandwidth utilization.
# -------------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector_types.h>

__global__ void leaky_relu_vec4_kernel(float* __restrict__ data, float negative_slope, int numel) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx + 3 < numel) {
        // Load, compute, and store float4 vectors
        float4 in_vec = reinterpret_cast<float4*>(data)[idx / 4];
        
        in_vec.x = in_vec.x > 0.0f ? in_vec.x : in_vec.x * negative_slope;
        in_vec.y = in_vec.y > 0.0f ? in_vec.y : in_vec.y * negative_slope;
        in_vec.z = in_vec.z > 0.0f ? in_vec.z : in_vec.z * negative_slope;
        in_vec.w = in_vec.w > 0.0f ? in_vec.w : in_vec.w * negative_slope;
        
        reinterpret_cast<float4*>(data)[idx / 4] = in_vec;
    } else {
        // Cleanup remaining elements
        for (int i = idx; i < numel; ++i) {
            float val = data[i];
            data[i] = val > 0.0f ? val : val * negative_slope;
        }
    }
}

void launch_leaky_relu(int numel, float* data, float negative_slope) {
    const int block_size = 1024;
    // Calculate grid size based on float4 processing (numel / 4)
    int grid_size = (numel / 4 + block_size - 1) / block_size;
    leaky_relu_vec4_kernel<<<grid_size, block_size>>>(data, negative_slope, numel);
}
"""

# -------------------------------------------------------------------------
# C++ binding: Expose the kernel to PyTorch
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void launch_leaky_relu(int numel, float* data, float negative_slope);

void leaky_relu_forward(torch::Tensor input, float negative_slope) {
    launch_leaky_relu(input.numel(), input.data_ptr<float>(), negative_slope);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("leaky_relu_forward", &leaky_relu_forward, "Vectorized In-place Leaky ReLU");
}
"""

# Compile the inline extension
leaky_relu_ext = load_inline(
    name='leaky_relu_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, negative_slope):
    """
    Optimized in-place functional Leaky ReLU. 
    Eliminates extra allocation to prevent VRAM swapping on the 2080 Ti.
    """
    if not x.is_contiguous():
        x = x.contiguous()
    
    # Perform in-place operation directly on the buffer of the input tensor
    leaky_relu_ext.leaky_relu_forward(x, float(negative_slope))
    return x

# Inputs matching the requirements for validation
batch_size = 4096
dim = 393216

def get_init_inputs():
    return []

def get_inputs():
    # Use torch memory contiguous allocation for performance
    x = torch.rand(batch_size, dim, device='cuda', dtype=torch.float32)
    return [x]
