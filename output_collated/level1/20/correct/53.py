# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_204858/code_28.py
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
# CUDA kernel – Heavily optimized for RTX 2080Ti using grid-stride loops
# -------------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void leaky_relu_vec4_kernel(float* __restrict__ data,
                                        float negative_slope,
                                        size_t numel) {
    // Each thread processes 4 elements per iteration
    size_t vec_numel = numel / 4;
    size_t idx = (blockIdx.x * blockDim.x + threadIdx.x);
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < vec_numel; i += stride) {
        float4* data_ptr = reinterpret_cast<float4*>(data);
        float4 in_vec = data_ptr[i];
        float4 out_vec;

        out_vec.x = in_vec.x > 0.0f ? in_vec.x : in_vec.x * negative_slope;
        out_vec.y = in_vec.y > 0.0f ? in_vec.y : in_vec.y * negative_slope;
        out_vec.z = in_vec.z > 0.0f ? in_vec.z : in_vec.z * negative_slope;
        out_vec.w = in_vec.w > 0.0f ? in_vec.w : in_vec.w * negative_slope;

        data_ptr[i] = out_vec;
    }

    // Handle leftover elements (if numel % 4 != 0)
    for (size_t i = idx * 4 + (vec_numel * 4); i < numel; i += stride) {
        float val = data[i];
        data[i] = val > 0.0f ? val : val * negative_slope;
    }
}

void launch_leaky_relu(at::Tensor& x, float negative_slope) {
    size_t numel = x.numel();
    float* data_ptr = x.data_ptr<float>();
    
    // Heuristic grid-block sizing for 2080Ti (Turing architecture)
    const int threads = 256;
    const int blocks = 1024; 
    
    leaky_relu_vec4_kernel<<<blocks, threads>>>(data_ptr, negative_slope, numel);
}
"""

# -------------------------------------------------------------------------
# C++ binding
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void launch_leaky_relu(at::Tensor& x, float negative_slope);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("leaky_relu_forward", &launch_leaky_relu, "Optimized Grid-Stride Leaky ReLU");
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
    Optimized in-place Leaky ReLU using a grid-stride kernel.
    """
    # 1. Ensure contiguity for coalesced float4 access
    if not x.is_contiguous():
        x = x.contiguous()
    
    # 2. Kernel launch
    leaky_relu_ext.leaky_relu_forward(x, float(negative_slope))
    return x
