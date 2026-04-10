# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_231959/code_18.py
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

# The optimization strategy uses shared memory as a staging area.
# Given that element-wise operations like Tanh have a high arithmetic intensity relative 
# to data movement (1 input, 1 output for 1 math operation), shared memory is often 
# used in kernels with data reuse (like stencil ops). 
# However, for simple point-wise kernels, the primary performance limit is global 
# memory bandwidth. We maintain float4-vectorized memory access to saturate the bus, 
# while ensuring the kernel structure is optimized for the RTX 2080Ti's architecture.

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// Tile size for shared memory usage. 1024 floats = 4KB, well within limits.
#define TILE_SIZE 1024

__global__ void tanh_kernel_vec4_optimized(const float* __restrict__ input, float* __restrict__ output, int numel) {
    // Shared memory buffer
    __shared__ float s_data[TILE_SIZE];

    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    int tid = threadIdx.x;
    
    // 1. Coalesced Loading into Shared Memory
    // We load a tile of data from global memory into shared memory.
    // Each thread loads 4 floats (float4)
    if (idx + 3 < numel) {
        float4 in_vec = reinterpret_cast<const float4*>(input)[(blockIdx.x * blockDim.x + tid)];
        
        // Store vector values into shared memory
        s_data[tid * 4 + 0] = in_vec.x;
        s_data[tid * 4 + 1] = in_vec.y;
        s_data[tid * 4 + 2] = in_vec.z;
        s_data[tid * 4 + 3] = in_vec.w;
        
        __syncthreads();

        // 2. Processing from Shared Memory
        float4 out_vec;
        out_vec.x = tanhf(s_data[tid * 4 + 0]);
        out_vec.y = tanhf(s_data[tid * 4 + 1]);
        out_vec.z = tanhf(s_data[tid * 4 + 2]);
        out_vec.w = tanhf(s_data[tid * 4 + 3]);

        // 3. Coalesced Write to Global Memory
        reinterpret_cast<float4*>(output)[(blockIdx.x * blockDim.x + tid)] = out_vec;
    } else {
        // Cleanup for remaining elements
        for (int i = idx; i < numel; ++i) {
            output[i] = tanhf(input[i]);
        }
    }
}

void launch_tanh_kernel(const torch::Tensor& input, torch::Tensor& output) {
    const int numel = input.numel();
    const int threads_per_block = 256;
    // Each block handles 256 * 4 = 1024 elements
    const int blocks = (numel / 1024 + 1);
    
    tanh_kernel_vec4_optimized<<<blocks, threads_per_block>>>(
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
    m.def("custom_tanh", &custom_tanh, "Vectorized CUDA tanh with shared memory buffering");
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

batch_size = 4096
dim = 393216

def get_init_inputs():
    return []

def get_inputs():
    x = torch.rand(batch_size, dim, device='cuda', dtype=torch.float32)
    return [x]
