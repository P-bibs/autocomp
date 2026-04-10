# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_004356/code_8.py
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

# Optimized CUDA kernel with pipelined computation and loop unrolling
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 1024  // 256 threads * 4 elements per thread
#define ELEMENTS_PER_THREAD 4

__global__ void tanh_kernel_optimized(const float* __restrict__ input, float* __restrict__ output, int numel) {
    extern __shared__ float shared_input[];
    
    int tid = threadIdx.x;
    int block_start_idx = blockIdx.x * TILE_SIZE;
    int global_idx = block_start_idx + tid * ELEMENTS_PER_THREAD;
    
    // Stage 1: Cooperative loading with vectorization
    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        int idx = global_idx + i;
        if (idx < numel) {
            shared_input[tid * ELEMENTS_PER_THREAD + i] = input[idx];
        }
    }
    
    __syncthreads();
    
    // Stage 2: Compute and write back directly (pipelined)
    // Each thread processes its 4 elements and writes directly to global memory
    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        int idx = global_idx + i;
        if (idx < numel) {
            // Read from shared memory, compute, and write to global memory
            float val = shared_input[tid * ELEMENTS_PER_THREAD + i];
            output[idx] = tanhf(val);
        }
    }
}

void launch_tanh_kernel(const torch::Tensor& input, torch::Tensor& output) {
    const int numel = input.numel();
    const int threads_per_block = 256;
    const int blocks = (numel + TILE_SIZE - 1) / TILE_SIZE;
    
    // Shared memory for input only (4KB instead of 8KB)
    const int shared_mem_size = TILE_SIZE * sizeof(float);
    
    tanh_kernel_optimized<<<blocks, threads_per_block, shared_mem_size>>>(
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
    m.def("custom_tanh", &custom_tanh, "Optimized pipelined CUDA tanh implementation");
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
