# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_202725/code_31.py
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
# CUDA kernel with grid-stride loop and optimized launch
# ----------------------------------------------------------------------
# Optimization notes:
# 1. Using grid-stride loops allows the kernel to be agnostic of the 
#    tensor size, reducing launch overhead.
# 2. __restrict__ pointers allow the compiler to assume no pointer 
#    aliasing, facilitating better load/store scheduling.
# 3. Dynamic SM-based grid sizing ensures saturation of the GPU 
#    without excessive command buffer overhead.
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void leaky_relu_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    float negative_slope,
    size_t n)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = tid; i < n; i += stride) {
        float val = input[i];
        output[i] = (val > 0.0f) ? val : __fmul_rn(val, negative_slope);
    }
}

// Host routine that chooses a balanced grid size
void leaky_relu_forward(torch::Tensor input, torch::Tensor output, float negative_slope) {
    const size_t n = input.numel();
    const int threads = 256;

    // Query number of multiprocessors to determine optimal grid size
    int device;
    cudaGetDevice(&device);
    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, device);
    
    // Launching with 16 blocks per SM is a standard heuristic for 
    // memory-bound elementwise operations to ensure high occupancy.
    int blocks = numSMs * 16;

    leaky_relu_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        negative_slope,
        n
    );
}
"""

# ----------------------------------------------------------------------
# C++ binding (PYBIND11)
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void leaky_relu_forward(torch::Tensor input, torch::Tensor output, float negative_slope);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("leaky_relu", &leaky_relu_forward, "Optimized Leaky ReLU forward kernel");
}
"""

# ----------------------------------------------------------------------
# Build the inline extension
# ----------------------------------------------------------------------
leaky_relu_ext = load_inline(
    name='leaky_relu_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, negative_slope):
    """
    Optimized functional_model using a grid-stride CUDA kernel.
    """
    # Ensure contiguous memory layout for optimal coalesced memory access
    if not x.is_contiguous():
        x = x.contiguous()
        
    if x.dtype != torch.float32:
        x = x.to(torch.float32)

    output = torch.empty_like(x)
    
    # Launch the custom grid-stride kernel
    leaky_relu_ext.leaky_relu(x, output, float(negative_slope))
    
    return output
