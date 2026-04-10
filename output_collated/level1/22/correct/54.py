# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_235928/code_16.py
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

# The grid-stride loop kernel provides better cache locality and allows 
# the kernel to scale independently of the input size, maximizing SM utilization.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void tanh_kernel_grid_stride(const float* __restrict__ input, float* __restrict__ output, int64_t numel) {
    // Grid-stride loop: each thread processes multiple 4-element chunks
    // This allows better occupancy and cache reuse.
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    
    // Convert current thread index to element index for float4
    int64_t idx = tid * 4;
    int64_t stride_elements = stride * 4;
    
    for (int64_t i = idx; i < numel; i += stride_elements) {
        if (i + 3 < numel) {
            float4 in_vec = reinterpret_cast<const float4*>(input + i)[0];
            float4 out_vec;
            out_vec.x = tanhf(in_vec.x);
            out_vec.y = tanhf(in_vec.y);
            out_vec.z = tanhf(in_vec.z);
            out_vec.w = tanhf(in_vec.w);
            reinterpret_cast<float4*>(output + i)[0] = out_vec;
        } else {
            // Cleanup for remaining elements (Tail logic)
            for (int64_t j = i; j < numel; ++j) {
                output[j] = tanhf(input[j]);
            }
        }
    }
}

void launch_tanh_kernel(const torch::Tensor& input, torch::Tensor& output) {
    const int64_t numel = input.numel();
    const int threads_per_block = 256;
    // We use a fixed occupancy-friendly grid size. 
    // 1024 blocks provides enough parallelism for a 2080Ti.
    const int blocks = 1024;
    
    tanh_kernel_grid_stride<<<blocks, threads_per_block>>>(
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
    m.def("custom_tanh", &custom_tanh, "Grid-stride Vectorized CUDA tanh");
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
    Computes tanh on the input using a optimized grid-stride CUDA kernel.
    """
    return tanh_ext.custom_tanh(x)

# Global variables for interface
batch_size = 4096
dim = 393216

def get_init_inputs():
    return []

def get_inputs():
    # Use GPU for performance as per requirements
    x = torch.rand(batch_size, dim, device='cuda', dtype=torch.float32)
    return [x]
