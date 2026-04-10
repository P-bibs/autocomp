# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_213956/code_5.py
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

# Optimized CUDA kernel using grid-stride loop for better scalability and occupancy
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void leaky_relu_grid_stride_kernel(const float* input, float* output, 
                                              float negative_slope, size_t n) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    // Grid-stride loop: each thread processes multiple elements
    for (size_t i = tid; i < n; i += stride) {
        float val = input[i];
        output[i] = (val > 0.0f) ? val : val * negative_slope;
    }
}

void leaky_relu_forward(torch::Tensor input, torch::Tensor output, float negative_slope) {
    const size_t n = input.numel();
    const int threads_per_block = 256;
    const int num_blocks = min(65535, (int)((n + threads_per_block - 1) / threads_per_block));
    
    leaky_relu_grid_stride_kernel<<<num_blocks, threads_per_block>>>(
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
    m.def("leaky_relu", &leaky_relu_forward, "Grid-stride Leaky ReLU forward kernel");
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
    Optimized functional_model using grid-stride CUDA kernel.
    Processes elements in a scalable way that adapts to input size.
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
