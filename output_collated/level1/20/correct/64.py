# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_211408/code_19.py
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

# CUDA Kernel: Vectorized Grid-Stride Kernel
# This combines coalesced float4 loads with grid-stride loops to maintain 
# high occupancy and max memory throughput on RTX 2080Ti.
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector_types.h>

__global__ void leaky_relu_kernel(const float* __restrict__ input, float* __restrict__ output, 
                                  float negative_slope, size_t n) {
    // Each thread processes 4 elements at a time via float4
    size_t vector_size = n / 4;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i = idx; i < vector_size; i += stride) {
        float4* vec_in = (float4*)(&input[i * 4]);
        float4* vec_out = (float4*)(&output[i * 4]);
        
        float4 val = *vec_in;
        
        val.x = (val.x > 0.0f) ? val.x : val.x * negative_slope;
        val.y = (val.y > 0.0f) ? val.y : val.y * negative_slope;
        val.z = (val.z > 0.0f) ? val.z : val.z * negative_slope;
        val.w = (val.w > 0.0f) ? val.w : val.w * negative_slope;
        
        *vec_out = val;
    }

    // Handle remainder
    size_t remainder_start = vector_size * 4;
    for (size_t i = remainder_start + idx; i < n; i += stride) {
        float val = input[i];
        output[i] = (val > 0.0f) ? val : val * negative_slope;
    }
}

void leaky_relu_forward(torch::Tensor input, torch::Tensor output, float negative_slope) {
    const size_t n = input.numel();
    const int threads = 256;
    // Aim for 32 blocks * SM count for high occupancy (RTX 2080Ti has 68 SMs)
    const int blocks = 2048; 
    
    leaky_relu_kernel<<<blocks, threads>>>(
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
    m.def("leaky_relu", &leaky_relu_forward, "Vectorized Coalesced Leaky ReLU");
}
"""

# Compile
leaky_relu_ext = load_inline(
    name='leaky_relu_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, negative_slope):
    """
    Optimized Leaky ReLU using vectorized float4 memory access.
    """
    x = x.contiguous().float()
    output = torch.empty_like(x)
    leaky_relu_ext.leaky_relu(x, output, float(negative_slope))
    return output

batch_size = 4096
dim = 393216

def get_inputs():
    return [torch.rand(batch_size, dim, device='cuda', dtype=torch.float32)]

def get_init_inputs():
    return []
