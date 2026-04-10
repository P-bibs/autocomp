# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_010905/code_10.py
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

# CUDA kernel with loop unrolling and increased elements per thread
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void tanh_kernel_unrolled(const float* __restrict__ input, float* __restrict__ output, int numel) {
    // Process 8 elements per thread (2x float4 vectors) with explicit loop unrolling
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    
    if (idx + 7 < numel) {
        // Load two float4 vectors (8 floats total)
        float4 in_vec0 = reinterpret_cast<const float4*>(input)[idx / 4];
        float4 in_vec1 = reinterpret_cast<const float4*>(input)[(idx + 4) / 4];
        
        // Compute tanh for all 8 elements with explicit unrolling
        float4 out_vec0;
        #pragma unroll
        for(int i = 0; i < 4; i++) {
            reinterpret_cast<float*>(&out_vec0)[i] = tanhf(reinterpret_cast<const float*>(&in_vec0)[i]);
        }
        
        float4 out_vec1;
        #pragma unroll
        for(int i = 0; i < 4; i++) {
            reinterpret_cast<float*>(&out_vec1)[i] = tanhf(reinterpret_cast<const float*>(&in_vec1)[i]);
        }
        
        // Store results
        reinterpret_cast<float4*>(output)[idx / 4] = out_vec0;
        reinterpret_cast<float4*>(output)[(idx + 4) / 4] = out_vec1;
    } else {
        // Handle remaining elements (cleanup)
        for (int i = idx; i < numel; ++i) {
            output[i] = tanhf(input[i]);
        }
    }
}

void launch_tanh_kernel(const torch::Tensor& input, torch::Tensor& output) {
    const int numel = input.numel();
    const int threads_per_block = 128;
    // Process 8 elements per thread, so we need 1/8 the original thread count
    const int blocks = (numel + threads_per_block * 8 - 1) / (threads_per_block * 8);
    
    tanh_kernel_unrolled<<<blocks, threads_per_block>>>(
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
    m.def("custom_tanh", &custom_tanh, "Unrolled CUDA tanh implementation");
}
"""

# Compile the extension with optimization flags
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
