# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_012911/code_30.py
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

# We provide __launch_bounds__ to guide the compiler in register allocation,
# which allows for higher occupancy, critical for memory-bound tanh operations.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ __launch_bounds__(256) void tanh_kernel_vec4(const float* __restrict__ input, float* __restrict__ output, int numel) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    
    if (idx + 3 < numel) {
        // Use float4 to achieve 128-bit memory transactions (maximum vector width)
        float4 in_vec = reinterpret_cast<const float4*>(input)[(idx) / 4];
        float4 out_vec;
        // tanhf is a fast native instruction supported by --use_fast_math
        out_vec.x = tanhf(in_vec.x);
        out_vec.y = tanhf(in_vec.y);
        out_vec.z = tanhf(in_vec.z);
        out_vec.w = tanhf(in_vec.w);
        reinterpret_cast<float4*>(output)[(idx) / 4] = out_vec;
    } else {
        // Handle remainder for non-aligned sizes
        for (int i = idx; i < numel; ++i) {
            output[i] = tanhf(input[i]);
        }
    }
}

void launch_tanh_kernel(const torch::Tensor& input, torch::Tensor& output) {
    const int numel = input.numel();
    const int threads = 256;
    // Utilize ceil division for grid size
    const int blocks = (numel / 4 + threads - 1) / threads;
    
    tanh_kernel_vec4<<<blocks, threads>>>(
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
    m.def("custom_tanh", &custom_tanh, "Optimized Vectorized CUDA tanh");
}
"""

# Compile the extension
# Using -arch=sm_75 for Turing/RTX 2080Ti and enabling fast math for throughput
tanh_ext = load_inline(
    name='tanh_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math', '-arch=sm_75'],
    with_cuda=True
)

def functional_model(x):
    """
    Applies the custom tanh operation.
    The implementation uses float4 vectorized loads, proper thread block bounds,
    and target-specific optimizations for the RTX 2080Ti.
    """
    return tanh_ext.custom_tanh(x)

# Interface configuration
batch_size = 4096
dim = 393216

def get_init_inputs():
    return []

def get_inputs():
    x = torch.rand(batch_size, dim, device='cuda', dtype=torch.float32)
    return [x]
