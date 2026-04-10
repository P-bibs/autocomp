# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_004356/code_20.py
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

# The optimization strategy focuses on maximizing throughput for memory-bound operators.
# 1. Use float4 for 128-bit memory transactions to reach peak bandwidth.
# 2. Implement grid-stride loops to allow the kernel to scale independently of hardware occupancy.
# 3. Strip away unnecessary shared memory, as tanh is register-bound and memory-latency dependent.

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void tanh_kernel_optimized(const float* __restrict__ input, float* __restrict__ output, int numel) {
    // Vectorized pointer access
    int vec_numel = numel / 4;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Process 4 elements per thread per iteration
    for (int i = idx; i < vec_numel; i += stride) {
        float4 in_vec = reinterpret_cast<const float4*>(input)[i];
        float4 out_vec;
        out_vec.x = tanhf(in_vec.x);
        out_vec.y = tanhf(in_vec.y);
        out_vec.z = tanhf(in_vec.z);
        out_vec.w = tanhf(in_vec.w);
        reinterpret_cast<float4*>(output)[i] = out_vec;
    }

    // Handle tail elements (if numel is not a multiple of 4)
    if (idx == 0) {
        for (int i = vec_numel * 4; i < numel; ++i) {
            output[i] = tanhf(input[i]);
        }
    }
}

void launch_tanh_kernel(const torch::Tensor& input, torch::Tensor& output) {
    const int numel = input.numel();
    const int threads = 256;
    // Heuristic: scale blocks based on available multiprocessors
    const int blocks = 1024; 
    
    tanh_kernel_optimized<<<blocks, threads>>>(
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
    m.def("custom_tanh", &custom_tanh, "Optimized global-memory tanh kernel");
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
    """
    Computes tanh of the input tensor using an optimized CUDA kernel.
    The kernel bypasses shared memory, utilizing float4 vectorization 
    and grid-stride loops to saturate GPU memory bandwidth.
    """
    return tanh_ext.custom_tanh(x)

# Interface requirements
batch_size = 4096
dim = 393216

def get_init_inputs():
    return []

def get_inputs():
    x = torch.rand(batch_size, dim, device='cuda', dtype=torch.float32)
    return [x]
