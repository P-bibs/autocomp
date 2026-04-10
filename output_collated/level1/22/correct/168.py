# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_012911/code_25.py
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

# We use a grid-stride loop pattern. This is robust, maximizes occupancy, 
# and allows the GPU scheduler to distribute work dynamically.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void tanh_kernel_optimized(const float* __restrict__ input, float* __restrict__ output, size_t n) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    // Process 4 elements at a time using float4 for vectorized memory access
    // This reduces the number of instructions and improves memory throughput
    for (size_t i = tid * 4; i + 3 < n; i += stride * 4) {
        float4* in_vec = (float4*)(input + i);
        float4* out_vec = (float4*)(output + i);
        
        float4 val = *in_vec;
        val.x = tanhf(val.x);
        val.y = tanhf(val.y);
        val.z = tanhf(val.z);
        val.w = tanhf(val.w);
        *out_vec = val;
    }
    
    // Handle remaining elements if n is not a multiple of 4 or the stride
    for (size_t i = tid; i < n; i += stride) {
        if ((i / 4) * 4 >= (n / 4) * 4) {
            output[i] = tanhf(input[i]);
        }
    }
}

void launch_tanh_kernel(const torch::Tensor& input, torch::Tensor& output) {
    const size_t n = input.numel();
    const int threads = 256;
    // Aim for ~1000+ blocks to ensure full occupancy on 68 SMs
    const int blocks = std::min((int)((n + (threads * 4) - 1) / (threads * 4)), 2048);
    
    tanh_kernel_optimized<<<blocks, threads>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        n
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void launch_tanh_kernel(const torch::Tensor& input, torch::Tensor& output);

torch::Tensor custom_tanh_batched(const torch::Tensor& input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    auto output = torch::empty_like(input);
    launch_tanh_kernel(input, output);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("custom_tanh_batched", &custom_tanh_batched, "Optimized float4 batched tanh");
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

# Global variables for interface
batch_size = 4096
dim = 393216

def get_init_inputs():
    return []

def get_inputs():
    # Ensure inputs are on GPU
    x = torch.rand(batch_size, dim, device='cuda', dtype=torch.float32)
    return [x]
