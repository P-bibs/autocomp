# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_012911/code_2.py
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

# Further optimized CUDA kernel using cooperative grid-stride loops and register blocking
# for maximum ALU-to-memory ratio on RTX 2080Ti

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define REG_BLOCK_SIZE 8

__global__ void tanh_kernel_coop_grid_stride(const float* __restrict__ input, float* __restrict__ output, int numel) {
    // Cooperative grid-stride loop with register blocking
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_stride = gridDim.x * blockDim.x;
    
    // Register blocking: process 8 float4 vectors per iteration (32 floats)
    // This increases ALU intensity and reduces memory bandwidth pressure
    float4 in_vecs[REG_BLOCK_SIZE];
    float4 out_vecs[REG_BLOCK_SIZE];
    
    for (int base_idx = tid * 4; base_idx < numel; base_idx += grid_stride * 4) {
        // Check if we have full register block to process
        if (base_idx + REG_BLOCK_SIZE * 4 <= numel) {
            // Vectorized load of 8 float4 vectors (128 floats)
            #pragma unroll
            for (int i = 0; i < REG_BLOCK_SIZE; i++) {
                in_vecs[i] = reinterpret_cast<const float4*>(input)[(base_idx + i * 4) / 4];
            }
            
            // Compute tanh on register block
            #pragma unroll
            for (int i = 0; i < REG_BLOCK_SIZE; i++) {
                out_vecs[i].x = tanhf(in_vecs[i].x);
                out_vecs[i].y = tanhf(in_vecs[i].y);
                out_vecs[i].z = tanhf(in_vecs[i].z);
                out_vecs[i].w = tanhf(in_vecs[i].w);
            }
            
            // Vectorized store of results
            #pragma unroll
            for (int i = 0; i < REG_BLOCK_SIZE; i++) {
                reinterpret_cast<float4*>(output)[(base_idx + i * 4) / 4] = out_vecs[i];
            }
        } else {
            // Tail handling: process remaining elements scalarly
            for (int i = base_idx; i < numel && i < base_idx + REG_BLOCK_SIZE * 4; i++) {
                output[i] = tanhf(input[i]);
            }
        }
    }
}

void launch_tanh_kernel(const torch::Tensor& input, torch::Tensor& output) {
    const int numel = input.numel();
    
    // Optimized configuration for RTX 2080Ti:
    // - Block size of 256 gives good occupancy (4 warps per SM)
    // - Limited grid size to avoid overhead while maintaining good parallelism
    const int threads_per_block = 256;
    // Ensure we launch enough blocks to keep all SMs busy
    const int blocks_per_grid = min(4 * 68, (numel / (threads_per_block * 4) + 1));
    
    tanh_kernel_coop_grid_stride<<<blocks_per_grid, threads_per_block>>>(
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
    m.def("custom_tanh", &custom_tanh, "Cooperative grid-stride loop with register blocking CUDA tanh implementation");
}
"""

# Compile the extension
tanh_ext = load_inline(
    name='tanh_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math', '-arch=compute_75', '-code=sm_75'],
    with_cuda=True
)

def functional_model(x):
    """
    Applies the custom tanh operation to the input tensor.
    The underlying CUDA implementation uses cooperative grid-stride loops with register
    blocking to maximize ALU-to-memory ratio and memory throughput on the RTX 2080Ti.
    """
    return tanh_ext.custom_tanh(x)

# Interface configuration
batch_size = 4096
dim = 393216

def get_init_inputs():
    return []

def get_inputs():
    # Input is on GPU as requested
    x = torch.rand(batch_size, dim, device='cuda', dtype=torch.float32)
    return [x]
