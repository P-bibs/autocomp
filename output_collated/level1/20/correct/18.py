# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_202725/code_18.py
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

# CUDA kernel with grid-stride loop for high-performance element-wise processing
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void leaky_relu_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float negative_slope,
    const int numel) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < numel; i += stride) {
        float val = input[i];
        output[i] = (val > 0.0f) ? val : (val * negative_slope);
    }
}

void launch_leaky_relu(
    const float* input,
    float* output,
    const float negative_slope,
    const int numel) {
    
    const int threads = 1024;
    // Calculate blocks to maximize occupancy, limited by hardware capacity
    const int blocks = (numel + threads - 1) / threads;
    const int grid_size = std::min(blocks, 65535); 
    
    leaky_relu_kernel<<<grid_size, threads>>>(input, output, negative_slope, numel);
}
"""

# C++ interface for binding
cpp_source = r"""
#include <torch/extension.h>

void launch_leaky_relu(const float* input, float* output, const float negative_slope, const int numel);

torch::Tensor fused_leaky_relu(torch::Tensor input, float negative_slope) {
    auto input_contig = input.contiguous();
    auto output = torch::empty_like(input_contig);
    
    launch_leaky_relu(
        input_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        negative_slope,
        (int)input_contig.numel()
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_leaky_relu", &fused_leaky_relu, "Fused Leaky ReLU CUDA kernel");
}
"""

# Compile the extension inline
fused_ext = load_inline(
    name='leaky_relu_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, negative_slope):
    """
    Optimized Leaky ReLU using custom CUDA kernel.
    """
    return fused_ext.fused_leaky_relu(x, negative_slope)

# Provided setup
batch_size = 4096
dim = 393216

def get_init_inputs():
    return []

def get_inputs():
    x = torch.rand(batch_size, dim, device='cuda')
    return [x]
