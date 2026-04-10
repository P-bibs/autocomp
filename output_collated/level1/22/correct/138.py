# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_010905/code_20.py
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

# The implementation uses a grid-stride loop to process input elements using float4 
# vectorization, which maximizes memory bandwidth efficiency. By decoupling the 
# number of threads from the problem size, we ensure high occupancy and portability
# across different GPU architectures.

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void tanh_grid_stride_kernel(const float* __restrict__ input, float* __restrict__ output, int numel) {
    // Calculate global index and stride
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Process float4 chunks to maximize memory throughput
    int i = tid * 4;
    for (; i + 3 < numel; i += stride * 4) {
        float4 in_vec = reinterpret_cast<const float4*>(input + i)[0];
        float4 out_vec;
        
        // Fast math tanhf approximation can be used if precision requirements allow
        out_vec.x = tanhf(in_vec.x);
        out_vec.y = tanhf(in_vec.y);
        out_vec.z = tanhf(in_vec.z);
        out_vec.w = tanhf(in_vec.w);
        
        reinterpret_cast<float4*>(output + i)[0] = out_vec;
    }
    
    // Cleanup/boundary handling for tensors not divisible by 4
    if (i < numel) {
        for (; i < numel; ++i) {
            output[i] = tanhf(input[i]);
        }
    }
}

void launch_tanh_kernel(const torch::Tensor& input, torch::Tensor& output) {
    const int numel = input.numel();
    const int threads = 256;
    
    // Calculate adaptive grid size: max blocks based on device multiprocessors
    // Using 1024 as a broad heuristic for standard GPU scaling
    int blocks = (numel + (threads * 4) - 1) / (threads * 4);
    blocks = (blocks > 1024) ? 1024 : blocks;
    
    tanh_grid_stride_kernel<<<blocks, threads>>>(
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
    m.def("custom_tanh", &custom_tanh, "Grid-stride vectorized tanh implementation");
}
"""

# Compile the JIT extension
tanh_ext = load_inline(
    name='tanh_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x):
    """
    Applies the custom CUDA tanh kernel to the input tensor.
    """
    return tanh_ext.custom_tanh(x)

# Setup parameters as specified
batch_size = 4096
dim = 393216

def get_inputs():
    # Returning input on GPU
    return [torch.rand(batch_size, dim, device='cuda', dtype=torch.float32)]
