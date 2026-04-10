# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_231959/code_15.py
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

# The CUDA kernel uses a grid-stride loop for scalability and float4 vectorization 
# for maximum bandwidth utilization. This approach ensures consistent performance 
# across different input sizes and hides memory latency through parallelism.

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void tanh_grid_stride_kernel(const float* __restrict__ input, float* __restrict__ output, int numel) {
    // Each thread processes 4 elements at once via float4 for memory alignment
    int stride = blockDim.x * gridDim.x * 4;
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    
    // Grid-stride loop: each thread iterates over the data until global completion
    for (int i = idx; i < numel; i += stride) {
        if (i + 3 < numel) {
            float4 in_vec = reinterpret_cast<const float4*>(input + i)[0];
            float4 out_vec;
            out_vec.x = tanhf(in_vec.x);
            out_vec.y = tanhf(in_vec.y);
            out_vec.z = tanhf(in_vec.z);
            out_vec.w = tanhf(in_vec.w);
            reinterpret_cast<float4*>(output + i)[0] = out_vec;
        } else {
            // Cleanup for remainder elements
            for (int k = i; k < numel; ++k) {
                output[k] = tanhf(input[k]);
            }
        }
    }
}

void launch_tanh_kernel(const torch::Tensor& input, torch::Tensor& output) {
    const int numel = input.numel();
    const int threads = 256;
    // Heuristic: Use a sufficient grid size to saturate the GPU
    // 1024 blocks provides 262,144 threads, suitable for large tensors
    const int blocks = 1024;
    
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
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    
    auto output = torch::empty_like(input);
    launch_tanh_kernel(input, output);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("custom_tanh", &custom_tanh, "Grid-stride vectorized tanh kernel");
}
"""

# Compile the extension inline
tanh_ext = load_inline(
    name='tanh_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x):
    """
    Applies the custom highly-optimized CUDA tanh.
    Ensure input is contiguous for float4 vectorization access.
    """
    if not x.is_contiguous():
        x = x.contiguous()
    return tanh_ext.custom_tanh(x)

# Global variables for testing purposes
batch_size = 4096
dim = 393216

def get_init_inputs():
    return []

def get_inputs():
    # Returns the input ready for the optimized kernel
    x = torch.rand(batch_size, dim, device='cuda', dtype=torch.float32)
    return [x]
