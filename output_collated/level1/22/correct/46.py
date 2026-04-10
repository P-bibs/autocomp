# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_235928/code_7.py
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

# CUDA kernel with aggressive loop unrolling for maximum instruction-level parallelism
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void tanh_kernel_vec4_unrolled(const float* __restrict__ input, float* __restrict__ output, int numel) {
    // Process 8 elements per thread (2 float4 vectors) for better ILP
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    
    // Main vectorized path: process 8 elements per thread
    if (idx + 7 < numel) {
        float4 in_vec1 = reinterpret_cast<const float4*>(input)[idx / 4];
        float4 in_vec2 = reinterpret_cast<const float4*>(input)[(idx + 4) / 4];
        
        float4 out_vec1;
        float4 out_vec2;
        
        // Unroll: apply tanhf to all 8 elements with maximum parallelism
        out_vec1.x = tanhf(in_vec1.x);
        out_vec1.y = tanhf(in_vec1.y);
        out_vec2.x = tanhf(in_vec2.x);
        out_vec2.y = tanhf(in_vec2.y);
        out_vec1.z = tanhf(in_vec1.z);
        out_vec1.w = tanhf(in_vec1.w);
        out_vec2.z = tanhf(in_vec2.z);
        out_vec2.w = tanhf(in_vec2.w);
        
        reinterpret_cast<float4*>(output)[idx / 4] = out_vec1;
        reinterpret_cast<float4*>(output)[(idx + 4) / 4] = out_vec2;
    } 
    // Secondary path: process 4 elements if remaining >= 4
    else if (idx + 3 < numel) {
        float4 in_vec = reinterpret_cast<const float4*>(input)[idx / 4];
        float4 out_vec;
        out_vec.x = tanhf(in_vec.x);
        out_vec.y = tanhf(in_vec.y);
        out_vec.z = tanhf(in_vec.z);
        out_vec.w = tanhf(in_vec.w);
        reinterpret_cast<float4*>(output)[idx / 4] = out_vec;
    } 
    // Cleanup path: unrolled loop for remaining elements
    else {
        #pragma unroll 4
        for (int i = idx; i < numel && i < idx + 4; ++i) {
            output[i] = tanhf(input[i]);
        }
    }
}

void launch_tanh_kernel(const torch::Tensor& input, torch::Tensor& output) {
    const int numel = input.numel();
    const int threads_per_block = 256;
    // Process 8 elements per thread, so we need 1/8 the blocks
    const int blocks = (numel + threads_per_block * 8 - 1) / (threads_per_block * 8);
    
    tanh_kernel_vec4_unrolled<<<blocks, threads_per_block>>>(
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
    m.def("custom_tanh", &custom_tanh, "Vectorized CUDA tanh with loop unrolling");
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
