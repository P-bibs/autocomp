# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_204858/code_20.py
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
# CUDA Kernel: Optimized for Memory Bandwidth
# -------------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void leaky_relu_kernel(float* __restrict__ data,
                                  float negative_slope,
                                  int numel) {
    // Each thread processes a float4 (4 contiguous floats) to saturate memory bus
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    
    if (idx + 3 < numel) {
        float4* data_vec = reinterpret_cast<float4*>(data);
        float4 val = data_vec[idx / 4];

        val.x = (val.x > 0.0f) ? val.x : val.x * negative_slope;
        val.y = (val.y > 0.0f) ? val.y : val.y * negative_slope;
        val.z = (val.z > 0.0f) ? val.z : val.z * negative_slope;
        val.w = (val.w > 0.0f) ? val.w : val.w * negative_slope;

        data_vec[idx / 4] = val;
    } else {
        // Unrolled tail handling for numel not divisible by 4
        #pragma unroll
        for (int i = idx; i < numel; ++i) {
            float val = data[i];
            data[i] = (val > 0.0f) ? val : val * negative_slope;
        }
    }
}

void launch_leaky_relu(torch::Tensor input, float negative_slope) {
    int numel = input.numel();
    int threads = 256;
    // Calculate blocks based on float4 processing
    int blocks = ((numel + 3) / 4 + threads - 1) / threads;
    
    leaky_relu_kernel<<<blocks, threads>>>(input.data_ptr<float>(), negative_slope, numel);
}
"""

# -------------------------------------------------------------------------
# C++ Binding
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void launch_leaky_relu(torch::Tensor input, float negative_slope);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("leaky_relu_forward", &launch_leaky_relu, "Vectorized Leaky ReLU (in-place)");
}
"""

# Compile the extension
leaky_relu_ext = load_inline(
    name='leaky_relu_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, negative_slope):
    """
    Optimized in-place Leaky ReLU using custom CUDA kernel.
    """
    if not x.is_contiguous():
        x = x.contiguous()
    
    # Execute optimized CUDA kernel
    leaky_relu_ext.leaky_relu_forward(x, float(negative_slope))
    
    return x
