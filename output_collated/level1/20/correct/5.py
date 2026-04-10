# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_201703/code_7.py
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
# CUDA source – hand‑tuned leaky‑ReLU kernel
# -------------------------------------------------------------------------
# The grid-stride loop ensures we can process arbitrary sizes while 
# staying within hardware launch constraints, maximizing occupancy.
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void leaky_relu_kernel(const float* __restrict__ input,
                                   float* __restrict__ output,
                                   const float negative_slope,
                                   const int64_t N) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    
    for (int64_t i = idx; i < N; i += stride) {
        // Use __ldg to hint read-only access (L1 cache)
        float x = __ldg(&input[i]);
        // Fast leaky-relu arithmetic
        output[i] = (x > 0.0f) ? x : (x * negative_slope);
    }
}

void leaky_relu_cuda(const torch::Tensor& input,
                     const float negative_slope,
                     torch::Tensor& output) {
    const int64_t N = input.numel();
    const int threads = 256;
    // Calculate grid size to saturate GPU (cap at 65535 or larger supported grid)
    const int blocks = (int)min((int64_t)65535, (N + threads - 1) / threads);

    leaky_relu_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        negative_slope, 
        N
    );
}
"""

# -------------------------------------------------------------------------
# C++ binding (PYBIND11)
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void leaky_relu_cuda(const torch::Tensor& input,
                     const float negative_slope,
                     torch::Tensor& output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("leaky_relu", &leaky_relu_cuda, "Leaky ReLU Custom CUDA Kernel");
}
"""

# Compile the extension once
leaky_relu_ext = load_inline(
    name='leaky_relu_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True,
)

def functional_model(x, *, negative_slope):
    """
    Optimized implementation of Leaky ReLU using a custom CUDA kernel.
    """
    # Allocate output tensor on the same device as input
    output = torch.empty_like(x)
    
    # Execute the compiled CUDA kernel
    leaky_relu_ext.leaky_relu(x, negative_slope, output)
    
    return output

def get_init_inputs():
    return []

def get_inputs():
    # Parameters provided in the prompt
    batch_size = 4096
    dim = 393216
    # Ensure tensor is on CUDA for the kernel to access it directly via data_ptr
    x = torch.rand(batch_size, dim, device='cuda', dtype=torch.float32)
    return [x]
