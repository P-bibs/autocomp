# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_001958/code_24.py
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

# Optimized CUDA kernel:
# 1. Removed Shared Memory: Bandwidth-bound kernels like tanh receive no benefit from shared memory.
# 2. Vectorized Loads/Stores: Uses float4 to maximize throughput per instruction.
# 3. Minimized Boundary Checks: Uses a simplified loop structure to maintain peak performance.
# 4. Increased Occupancy: 512 threads/block helps hide memory latency better than 256.

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void tanh_kernel_direct(const float* __restrict__ input, float* __restrict__ output, int numel) {
    // Each thread processes 4 elements (128 bits)
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    
    // Main loop for vectorized processing
    if (idx + 3 < numel) {
        // Coalesced float4 load
        float4 in_vec = reinterpret_cast<const float4*>(input)[(idx >> 2)];
        
        // Fast tanh implementation using compiler's fast math and math library intrinsics
        float4 out_vec;
        out_vec.x = tanhf(in_vec.x);
        out_vec.y = tanhf(in_vec.y);
        out_vec.z = tanhf(in_vec.z);
        out_vec.w = tanhf(in_vec.w);
        
        // Coalesced float4 store
        reinterpret_cast<float4*>(output)[(idx >> 2)] = out_vec;
    } else {
        // Scalar boundary handling
        for (int i = 0; i < 4; i++) {
            if (idx + i < numel) {
                output[idx + i] = tanhf(input[idx + i]);
            }
        }
    }
}

void launch_tanh_kernel(const torch::Tensor& input, torch::Tensor& output) {
    const int numel = input.numel();
    const int threads_per_block = 512;
    // Each thread processes 4 floats
    const int blocks = (numel + (threads_per_block * 4) - 1) / (threads_per_block * 4);
    
    tanh_kernel_direct<<<blocks, threads_per_block>>>(
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
    m.def("custom_tanh", &custom_tanh, "Direct global memory CUDA tanh implementation");
}
"""

# Compile the extension with aggressive optimizations
tanh_ext = load_inline(
    name='tanh_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math', '--ptxas-options=-v'],
    with_cuda=True
)

def functional_model(x):
    """
    Computes tanh elementwise using the optimized custom CUDA kernel.
    """
    return tanh_ext.custom_tanh(x)

# Setup inputs for evaluation
batch_size = 4096
dim = 393216

def get_init_inputs():
    return []

def get_inputs():
    # Use pin_memory or direct GPU placement for performance parity
    x = torch.rand(batch_size, dim, device='cuda', dtype=torch.float32)
    return [x]
