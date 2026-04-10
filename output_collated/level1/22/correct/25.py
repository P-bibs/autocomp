# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_233710/code_8.py
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

# Optimized CUDA kernel: eliminate output shared memory buffer and reduce synchronization
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 1024  // 256 threads * 4 elements per thread

__global__ void tanh_kernel_shared(const float* __restrict__ input, float* __restrict__ output, int numel) {
    extern __shared__ float shared_input[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * TILE_SIZE + threadIdx.x * 4;
    
    // Cooperative loading of data into shared memory
    if (idx + 3 < numel) {
        float4 in_vec = reinterpret_cast<const float4*>(input)[idx / 4];
        reinterpret_cast<float4*>(shared_input)[tid] = in_vec;
    } else {
        // Handle boundary elements
        for (int i = 0; i < 4; i++) {
            if (idx + i < numel) {
                shared_input[tid * 4 + i] = input[idx + i];
            }
        }
    }
    
    __syncthreads();
    
    // Process data directly: read from shared, compute tanh, write to global
    // No intermediate output buffer in shared memory
    if (idx + 3 < numel) {
        float4 in_vec = reinterpret_cast<const float4*>(shared_input)[tid];
        float4 out_vec;
        out_vec.x = tanhf(in_vec.x);
        out_vec.y = tanhf(in_vec.y);
        out_vec.z = tanhf(in_vec.z);
        out_vec.w = tanhf(in_vec.w);
        // Write directly to global memory
        reinterpret_cast<float4*>(output)[idx / 4] = out_vec;
    } else {
        // Handle boundary elements
        for (int i = 0; i < 4; i++) {
            if (idx + i < numel) {
                output[idx + i] = tanhf(shared_input[tid * 4 + i]);
            }
        }
    }
}

void launch_tanh_kernel(const torch::Tensor& input, torch::Tensor& output) {
    const int numel = input.numel();
    const int threads_per_block = 256;
    const int blocks = (numel + TILE_SIZE - 1) / TILE_SIZE;
    
    // Shared memory only for input (TILE_SIZE floats = 4KB)
    const int shared_mem_size = TILE_SIZE * sizeof(float);
    
    tanh_kernel_shared<<<blocks, threads_per_block, shared_mem_size>>>(
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
    m.def("custom_tanh", &custom_tanh, "Optimized shared memory CUDA tanh: reduced sync points and shared memory");
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
