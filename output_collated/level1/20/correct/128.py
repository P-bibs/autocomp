# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_222822/code_18.py
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

# The provided CUDA kernel utilizes a grid-stride loop combined with float4 vectorization.
# Using __restrict__ and appropriate block/grid sizing ensures maximum memory throughput
# on the RTX 2080Ti, fully leveraging the hardware's memory bandwidth.

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector_types.h>

__global__ void leaky_relu_grid_stride_kernel(const float* __restrict__ input, 
                                              float* __restrict__ output, 
                                              float negative_slope, 
                                              size_t n) {
    // Each thread processes 4 elements per iteration using float4 vectorization.
    // The grid-stride loop ensures that all SMs are saturated regardless of input size.
    size_t i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    size_t stride = (size_t)gridDim.x * blockDim.x * 4;
    
    for (; i < n; i += stride) {
        if (i + 4 <= n) {
            float4 vals = *reinterpret_cast<const float4*>(&input[i]);
            float4 res;
            res.x = (vals.x > 0.0f) ? vals.x : vals.x * negative_slope;
            res.y = (vals.y > 0.0f) ? vals.y : vals.y * negative_slope;
            res.z = (vals.z > 0.0f) ? vals.z : vals.z * negative_slope;
            res.w = (vals.w > 0.0f) ? vals.w : vals.w * negative_slope;
            *reinterpret_cast<float4*>(&output[i]) = res;
        } else {
            // Remainder handling
            for (size_t j = i; j < n; ++j) {
                float val = input[j];
                output[j] = (val > 0.0f) ? val : val * negative_slope;
            }
        }
    }
}

void leaky_relu_forward(torch::Tensor input, torch::Tensor output, float negative_slope) {
    const size_t n = input.numel();
    const int threads_per_block = 256;
    // Heuristic for RTX 2080Ti occupancy: 68 SMs. 
    // We launch enough blocks to cover the grid with sufficient parallelism.
    const int blocks = std::min((int)((n + 1023) / 1024), 2048);
    
    leaky_relu_grid_stride_kernel<<<blocks, threads_per_block>>>(
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
    Optimized functional_model entry point.
    Ensures input alignment and calls the JIT-compiled CUDA kernel.
    """
    if x.dtype != torch.float32:
        x = x.to(torch.float32)
    if not x.is_contiguous():
        x = x.contiguous()
    
    output = torch.empty_like(x)
    leaky_relu_ext.leaky_relu(x, output, float(negative_slope))
    return output

# Constants for evaluation
batch_size = 4096
dim = 393216

def get_init_inputs():
    return []

def get_inputs():
    x = torch.rand(batch_size, dim, device='cuda', dtype=torch.float32)
    return [x]
