# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_185153/code_25.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = []
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = []
REQUIRED_FLAT_STATE_NAMES = []


class ModelNew(nn.Module):
    """
    Simple model that performs a ReLU activation.
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

# The CUDA kernel implements a grid-stride loop with vectorized float4 loads.
# It ensures all threads stay busy regardless of input size, maximizing
# memory bandwidth utilization on the RTX 2080Ti.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void relu_grid_stride_kernel(const float* __restrict__ input, float* __restrict__ output, const size_t num_elements) {
    const size_t stride = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total_threads = blockDim.x * gridDim.x;
    const size_t vec_size = num_elements / 4;

    // Process elements in float4 chunks
    for (size_t i = stride; i < vec_size; i += total_threads) {
        float4 in_vec = reinterpret_cast<const float4*>(input)[i];
        float4 out_vec;
        
        out_vec.x = fmaxf(0.0f, in_vec.x);
        out_vec.y = fmaxf(0.0f, in_vec.y);
        out_vec.z = fmaxf(0.0f, in_vec.z);
        out_vec.w = fmaxf(0.0f, in_vec.w);
        
        reinterpret_cast<float4*>(output)[i] = out_vec;
    }

    // Handle remaining scalar elements if num_elements is not a multiple of 4
    // Using a simple grid-stride loop for the remainder
    const size_t remainder_start = vec_size * 4;
    for (size_t i = remainder_start + stride; i < num_elements; i += total_threads) {
        output[i] = fmaxf(0.0f, input[i]);
    }
}

void launch_relu_kernel(const float* input, float* output, const size_t num_elements) {
    const int threads_per_block = 256;
    // Aiming for high occupancy: 1024 total threads per block/grid structure
    // 2080Ti has 68 SMs, 768 blocks is a robust choice to saturate the GPU
    const int num_blocks = 768; 
    
    relu_grid_stride_kernel<<<num_blocks, threads_per_block>>>(input, output, num_elements);
}
"""

cpp_source = r"""
#include <torch/extension.h>

void launch_relu_kernel(const float* input, float* output, const size_t num_elements);

void relu_forward(torch::Tensor input, torch::Tensor output) {
    launch_relu_kernel(input.data_ptr<float>(), output.data_ptr<float>(), input.numel());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("relu_forward", &relu_forward, "Vectorized ReLU forward pass with grid-stride loops");
}
"""

# Compile the extension
_relu_ext = load_inline(
    name='relu_op_optimized',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x):
    """
    Optimized ReLU function using a custom CUDA kernel with grid-stride loops
    and vectorized float4 memory access.
    """
    if not x.is_contiguous():
        x = x.contiguous()
    
    output = torch.empty_like(x)
    _relu_ext.relu_forward(x, output)
    return output

# Inputs for performance/correctness checks
batch_size = 4096
dim = 393216

def get_init_inputs():
    return []

def get_inputs():
    x = torch.rand(batch_size, dim, device='cuda', dtype=torch.float32)
    return [x]
