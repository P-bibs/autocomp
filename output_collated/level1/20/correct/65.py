# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_211408/code_21.py
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

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector_types.h>

// Vectorized Leaky ReLU kernel using float4 to saturate memory bandwidth
__global__ void leaky_relu_vec4_kernel(const float* __restrict__ input, float* __restrict__ output, float negative_slope, int numel_vec4) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < numel_vec4) {
        float4 in_vec = reinterpret_cast<const float4*>(input)[idx];
        float4 out_vec;
        
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            float val = (&in_vec.x)[i];
            (&out_vec.x)[i] = (val > 0) ? val : val * negative_slope;
        }
        
        reinterpret_cast<float4*>(output)[idx] = out_vec;
    }
}

// Cleanup kernel for remaining elements
__global__ void leaky_relu_cleanup_kernel(const float* __restrict__ input, float* __restrict__ output, float negative_slope, int start_idx, int numel) {
    int idx = start_idx + (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < numel) {
        float val = input[idx];
        output[idx] = (val > 0) ? val : val * negative_slope;
    }
}

void launch_leaky_relu(int numel, float* input, float* output, float negative_slope) {
    const int block_size = 256;
    int numel_vec4 = numel / 4;
    
    if (numel_vec4 > 0) {
        int grid_size = (numel_vec4 + block_size - 1) / block_size;
        leaky_relu_vec4_kernel<<<grid_size, block_size>>>(input, output, negative_slope, numel_vec4);
    }
    
    int remaining = numel % 4;
    if (remaining > 0) {
        int start_idx = numel - remaining;
        leaky_relu_cleanup_kernel<<<1, remaining>>>(input, output, negative_slope, start_idx, numel);
    }
}
"""

cpp_source = r"""
#include <torch/extension.h>

void launch_leaky_relu(int numel, float* input, float* output, float negative_slope);

void leaky_relu_forward(torch::Tensor input, torch::Tensor output, float negative_slope) {
    launch_leaky_relu(input.numel(), input.data_ptr<float>(), output.data_ptr<float>(), negative_slope);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("leaky_relu_forward", &leaky_relu_forward, "Vectorized Leaky ReLU");
}
"""

leaky_relu_ext = load_inline(
    name='leaky_relu_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, negative_slope):
    """
    Optimized Leaky ReLU:
    1. Uses float4 vectorization to maximize bus utilization on the 2080Ti.
    2. Separates main workload from cleanup to eliminate warp divergence.
    3. Guarantees coalesced memory access via float4 structs.
    """
    if not x.is_contiguous():
        x = x.contiguous()
    output = torch.empty_like(x)
    leaky_relu_ext.leaky_relu_forward(x, output, float(negative_slope))
    return output
