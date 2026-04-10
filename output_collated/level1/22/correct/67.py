# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_001958/code_10.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = []
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = []
REQUIRED_FLAT_STATE_NAMES = []


class ModelNew(nn.Module):
    """
    Simple model that performs a Tanh activation.
    """

    def __init__(self):
        super(ModelNew, self).__init__()

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

# CUDA kernel with divergence-free vectorization
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void tanh_kernel_nodiv(const float* __restrict__ input, float* __restrict__ output, int numel) {
    // Process 4 elements per thread with no divergence
    int base_idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    
    // All threads execute the same path - no branching
    float4 in_vec;
    in_vec.x = (base_idx < numel) ? input[base_idx] : 0.0f;
    in_vec.y = (base_idx + 1 < numel) ? input[base_idx + 1] : 0.0f;
    in_vec.z = (base_idx + 2 < numel) ? input[base_idx + 2] : 0.0f;
    in_vec.w = (base_idx + 3 < numel) ? input[base_idx + 3] : 0.0f;
    
    // Apply tanh to all elements
    float4 out_vec;
    out_vec.x = tanhf(in_vec.x);
    out_vec.y = tanhf(in_vec.y);
    out_vec.z = tanhf(in_vec.z);
    out_vec.w = tanhf(in_vec.w);
    
    // Write back with predicate - no divergence, just predicated writes
    if (base_idx < numel) output[base_idx] = out_vec.x;
    if (base_idx + 1 < numel) output[base_idx + 1] = out_vec.y;
    if (base_idx + 2 < numel) output[base_idx + 2] = out_vec.z;
    if (base_idx + 3 < numel) output[base_idx + 3] = out_vec.w;
}

void launch_tanh_kernel(const torch::Tensor& input, torch::Tensor& output) {
    const int numel = input.numel();
    const int threads_per_block = 256;
    const int blocks = (numel + threads_per_block * 4 - 1) / (threads_per_block * 4);
    
    tanh_kernel_nodiv<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        numel
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void launch_tanh_kernel(const torch::Tensor& input, torch::Tensor& output);

torch::Tensor custom_tanh(const torch::Tensor& input) {
    auto output = torch::empty_like(input);
    launch_tanh_kernel(input, output);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("custom_tanh", &custom_tanh, "Divergence-free vectorized CUDA tanh implementation");
}
"""

# Compile the extension
tanh_ext = load_inline(
    name='tanh_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x):
    return tanh_ext.custom_tanh(x)

# Global variables for interface requirements
batch_size = 4096
dim = 393216

def get_init_inputs():
    return []

def get_inputs():
    # Ensure inputs are on GPU as per requirement 6/7
    x = torch.rand(batch_size, dim, device='cuda', dtype=torch.float32)
    return [x]
