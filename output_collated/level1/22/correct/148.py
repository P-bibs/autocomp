# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_012911/code_1.py
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

#define TILE_SIZE 256
#define ELEMENTS_PER_THREAD 4

__global__ void tanh_kernel_batched(const float* __restrict__ input, float* __restrict__ output, int batch_size, int dim) {
    // Shared memory for input data
    __shared__ float shared_input[TILE_SIZE * ELEMENTS_PER_THREAD];
    
    int row = blockIdx.y;
    int tid = threadIdx.x;
    int elements_per_block = blockDim.x * ELEMENTS_PER_THREAD;
    int start_col = blockIdx.x * elements_per_block;
    
    if (row < batch_size) {
        // Load data into shared memory in a coalesced manner
        const float* row_input = input + row * dim;
        float* row_output = output + row * dim;
        
        // Each thread loads 4 consecutive elements
        int global_idx = start_col + tid * ELEMENTS_PER_THREAD;
        
        // Check bounds and load data
        if (global_idx + 3 < dim) {
            // Vectorized load
            float4* shared_ptr = reinterpret_cast<float4*>(&shared_input[tid * ELEMENTS_PER_THREAD]);
            const float4* global_ptr = reinterpret_cast<const float4*>(&row_input[global_idx]);
            *shared_ptr = *global_ptr;
        } else {
            // Handle boundary conditions
            for (int i = 0; i < ELEMENTS_PER_THREAD && global_idx + i < dim; i++) {
                shared_input[tid * ELEMENTS_PER_THREAD + i] = row_input[global_idx + i];
            }
        }
        
        __syncthreads();
        
        // Process data from shared memory
        if (global_idx + 3 < dim) {
            // Vectorized processing
            float4 in_vec = *reinterpret_cast<float4*>(&shared_input[tid * ELEMENTS_PER_THREAD]);
            float4 out_vec;
            out_vec.x = tanhf(in_vec.x);
            out_vec.y = tanhf(in_vec.y);
            out_vec.z = tanhf(in_vec.z);
            out_vec.w = tanhf(in_vec.w);
            *reinterpret_cast<float4*>(&row_output[global_idx]) = out_vec;
        } else {
            // Handle boundary conditions
            for (int i = 0; i < ELEMENTS_PER_THREAD && global_idx + i < dim; i++) {
                row_output[global_idx + i] = tanhf(shared_input[tid * ELEMENTS_PER_THREAD + i]);
            }
        }
    }
}

void launch_tanh_kernel(const torch::Tensor& input, torch::Tensor& output) {
    const int batch_size = input.size(0);
    const int dim = input.size(1);
    
    // Threads per block
    dim3 threads(TILE_SIZE);
    // Grid: width covers dimension, height covers batch_size
    dim3 blocks((dim + threads.x * ELEMENTS_PER_THREAD - 1) / (threads.x * ELEMENTS_PER_THREAD), batch_size);
    
    tanh_kernel_batched<<<blocks, threads>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        batch_size, 
        dim
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
    m.def("custom_tanh_batched", &custom_tanh_batched, "Batched optimized tanh with shared memory");
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
