# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_220342/code_18.py
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

# CUDA Kernel for In-Place Vectorized Leaky ReLU
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector_types.h>

__global__ void leaky_relu_inplace_kernel(float* data, float negative_slope, size_t n) {
    size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    
    if (idx + 4 <= n) {
        float4 vals = *reinterpret_cast<float4*>(&data[idx]);
        
        // Compute in-place
        vals.x = (vals.x > 0.0f) ? vals.x : vals.x * negative_slope;
        vals.y = (vals.y > 0.0f) ? vals.y : vals.y * negative_slope;
        vals.z = (vals.z > 0.0f) ? vals.z : vals.z * negative_slope;
        vals.w = (vals.w > 0.0f) ? vals.w : vals.w * negative_slope;
        
        *reinterpret_cast<float4*>(&data[idx]) = vals;
    } else {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            size_t element_idx = idx + i;
            if (element_idx < n) {
                float val = data[element_idx];
                data[element_idx] = (val > 0.0f) ? val : val * negative_slope;
            }
        }
    }
}

void leaky_relu_inplace_forward(torch::Tensor input, float negative_slope) {
    const size_t n = input.numel();
    const int threads = 256;
    const int blocks = (n + (threads * 4) - 1) / (threads * 4);
    
    leaky_relu_inplace_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        negative_slope,
        n
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void leaky_relu_inplace_forward(torch::Tensor input, float negative_slope);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("leaky_relu_inplace", &leaky_relu_inplace_forward, "In-place Vectorized Leaky ReLU");
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
    Optimized functional_model:
    1. Performs in-place transformation to avoid extra allocation.
    2. Ensures memory is contiguous to maximize throughput.
    3. Uses 128-bit float4 loads/stores to saturate memory bandwidth.
    """
    # Force float32 and contiguous memory for coalesced 128-bit access
    if x.dtype != torch.float32:
        x = x.to(torch.float32)
    if not x.is_contiguous():
        x = x.contiguous()
    
    # Launch in-place kernel
    leaky_relu_ext.leaky_relu_inplace(x, float(negative_slope))
    return x

# Constants for evaluation
batch_size = 4096
dim = 393216

def get_init_inputs():
    return []

def get_inputs():
    return [torch.rand(batch_size, dim, device='cuda', dtype=torch.float32)]
