# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_001958/code_16.py
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

# Optimization: Direct memory streaming with grid-stride loops for better occupancy
# and reduced overhead. float4 vectorization ensures coalesced memory access
# without the latency/resource penalty of manual shared memory staging.

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void tanh_kernel_optimized(const float* __restrict__ input, float* __restrict__ output, int numel) {
    // Grid-stride loop pattern: 
    // This allows the kernel to handle arbitrary sizes while maintaining
    // coalesced memory access and keeping registers busy.
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    int stride = blockDim.x * gridDim.x * 4;

    for (; idx < (numel / 4) * 4; idx += stride) {
        float4 in_vec = reinterpret_cast<const float4*>(input)[idx / 4];
        float4 out_vec;
        out_vec.x = tanhf(in_vec.x);
        out_vec.y = tanhf(in_vec.y);
        out_vec.z = tanhf(in_vec.z);
        out_vec.w = tanhf(in_vec.w);
        reinterpret_cast<float4*>(output)[idx / 4] = out_vec;
    }

    // Handle trailing elements if numel is not a multiple of 4
    // Using a simple conditional check for the remaining portion
    int remaining_start = (numel / 4) * 4;
    for (int i = remaining_start + (blockIdx.x * blockDim.x + threadIdx.x); i < numel; i += blockDim.x * gridDim.x) {
        output[i] = tanhf(input[i]);
    }
}

void launch_tanh_kernel(const torch::Tensor& input, torch::Tensor& output) {
    const int numel = input.numel();
    // Using 256 threads per block is generally optimal for memory-bound ops
    const int threads = 256;
    // Cap blocks to prevent grid size from exceeding hardware limits for very large tensors
    const int blocks = min(65535, (numel / 4 + threads - 1) / threads);

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
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    auto output = torch::empty_like(input);
    launch_tanh_kernel(input, output);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("custom_tanh", &custom_tanh, "Optimized CUDA tanh implementation");
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
    # Ensure the input is contiguous for optimal memory access
    return tanh_ext.custom_tanh(x.contiguous())

# Global variables for interface requirements
batch_size = 4096
dim = 393216

def get_init_inputs():
    return []

def get_inputs():
    # Ensure inputs are on GPU
    x = torch.rand(batch_size, dim, device='cuda', dtype=torch.float32)
    return [x]
