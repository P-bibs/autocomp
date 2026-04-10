# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_012911/code_24.py
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

# Optimized CUDA kernel using grid-stride loops and vectorized float4 memory access.
# This approach eliminates the overhead of shared memory, increases occupancy,
# and maximizes memory bandwidth throughput on the RTX 2080Ti.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void tanh_kernel_gridstride(const float* __restrict__ input, float* __restrict__ output, int numel) {
    // Grid-stride loop allows this kernel to scale to any tensor size while 
    // keeping occupancy constant.
    const int stride = blockDim.x * gridDim.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread processes 4 elements at a time via float4 vectorization.
    // Memory access is coalesced as threads access adjacent float4 blocks.
    while (idx * 4 < numel) {
        int pos = idx * 4;
        
        // Standard optimized path for full 4-element vectors
        if (pos + 3 < numel) {
            float4 in_vec = reinterpret_cast<const float4*>(input)[idx];
            float4 out_vec;
            out_vec.x = tanhf(in_vec.x);
            out_vec.y = tanhf(in_vec.y);
            out_vec.z = tanhf(in_vec.z);
            out_vec.w = tanhf(in_vec.w);
            reinterpret_cast<float4*>(output)[idx] = out_vec;
        } else {
            // Handle tail elements without branching warp divergence where possible
            for (int i = 0; i < 4; ++i) {
                if (pos + i < numel) {
                    output[pos + i] = tanhf(input[pos + i]);
                }
            }
        }
        idx += stride;
    }
}

void launch_tanh_kernel(const torch::Tensor& input, torch::Tensor& output) {
    const int numel = input.numel();
    const int threads_per_block = 256;
    // Heuristic: Occupancy is maximized by keeping enough blocks to saturate 
    // the SMs of the RTX 2080Ti (68 SMs). 
    // Dividing by 4 because each thread handles 4 elements.
    const int blocks = std::min((numel + (threads_per_block * 4) - 1) / (threads_per_block * 4), 1024);
    
    tanh_kernel_gridstride<<<blocks, threads_per_block>>>(
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
    m.def("custom_tanh", &custom_tanh, "Grid-stride loop optimized CUDA tanh");
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
    """
    Applies the custom tanh CUDA kernel.
    Input x is expected to be on GPU.
    """
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
