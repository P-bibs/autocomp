# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_012911/code_9.py
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

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// Optimized 1D hierarchical kernel with coalesced memory access
__global__ void tanh_kernel_1d_coalesced(
    const float* __restrict__ input, 
    float* __restrict__ output, 
    int total_elements) {
    
    // Calculate global thread index
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    
    // Process elements with grid-stride loop for better load balancing
    for (int i = tid * 4; i < total_elements; i += stride * 4) {
        if (i + 3 < total_elements) {
            // Vectorized path: process 4 elements at once
            const float4* in_vec_ptr = reinterpret_cast<const float4*>(input + i);
            float4* out_vec_ptr = reinterpret_cast<float4*>(output + i);
            
            float4 in_vec = *in_vec_ptr;
            float4 out_vec;
            out_vec.x = tanhf(in_vec.x);
            out_vec.y = tanhf(in_vec.y);
            out_vec.z = tanhf(in_vec.z);
            out_vec.w = tanhf(in_vec.w);
            *out_vec_ptr = out_vec;
        } else {
            // Scalar path: handle remaining elements
            for (int j = 0; j < 4 && i + j < total_elements; ++j) {
                output[i + j] = tanhf(input[i + j]);
            }
            break;
        }
    }
}

void launch_tanh_kernel(const torch::Tensor& input, torch::Tensor& output) {
    const int total_elements = input.numel();
    
    // Optimize for RTX 2080Ti (68 SMs)
    // Use 256 threads per block (standard for good occupancy)
    const int threads_per_block = 256;
    // Aim for ~10-15 blocks per SM => 68*12 = ~816 blocks
    const int target_blocks = 8192; // Round up to next power of 2 for efficiency
    
    // Calculate actual number of blocks needed
    const int elements_per_block = threads_per_block * 4; // Each thread processes 4 elements
    const int required_blocks = (total_elements + elements_per_block - 1) / elements_per_block;
    const int blocks = min(target_blocks, required_blocks);
    
    tanh_kernel_1d_coalesced<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        total_elements
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void launch_tanh_kernel(const torch::Tensor& input, torch::Tensor& output);

torch::Tensor custom_tanh_batched(const torch::Tensor& input) {
    auto output = torch::empty_like(input);
    launch_tanh_kernel(input, output);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("custom_tanh_batched", &custom_tanh_batched, "Highly optimized 1D coalesced tanh");
}
"""

# Compile the extension
tanh_ext_opt = load_inline(
    name='tanh_ext_opt',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x):
    return tanh_ext_opt.custom_tanh_batched(x)

# Global variables for interface requirements
batch_size = 4096
dim = 393216

def get_init_inputs():
    return []

def get_inputs():
    # Ensure inputs are on GPU as per requirement 6/7
    x = torch.rand(batch_size, dim, device='cuda', dtype=torch.float32)
    return [x]
