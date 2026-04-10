# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_184018/code_23.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = []
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = []
REQUIRED_FLAT_STATE_NAMES = []


class ModelNew(nn.Module):
    """
    Simple model that performs a ReLU activation.
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

# -------------------------------------------------------------------------
# CUDA kernel – Vectorized ReLU using float4 (128-bit memory access)
# -------------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <algorithm>

__global__ void relu_kernel_vec(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int64_t num_elements)
{
    // Total number of elements we can process as float4
    const int64_t vec_num = num_elements / 4;
    const int64_t stride = (int64_t)blockDim.x * gridDim.x;
    
    // Process float4 chunks
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    
    for (int64_t i = idx; i < vec_num; i += stride) {
        float4 in = reinterpret_cast<const float4*>(input)[i];
        float4 out;
        out.x = fmaxf(0.0f, in.x);
        out.y = fmaxf(0.0f, in.y);
        out.z = fmaxf(0.0f, in.z);
        out.w = fmaxf(0.0f, in.w);
        reinterpret_cast<float4*>(output)[i] = out;
    }
    
    // Handle remaining tail elements for the last few threads
    // Only threads that would process the last remainder logic need to check
    if (idx == 0) {
        for (int64_t i = vec_num * 4; i < num_elements; ++i) {
            output[i] = fmaxf(0.0f, input[i]);
        }
    }
}

void relu_forward_cuda_vec(const torch::Tensor& input, torch::Tensor& output) {
    const int64_t num_elements = input.numel();
    const int threads = 256;
    // We process num_elements / 4 chunks
    const int64_t vec_num = num_elements / 4;
    const int blocks = (int)std::min((int64_t)65535, (vec_num + threads - 1) / threads);
    
    if (blocks > 0) {
        relu_kernel_vec<<<blocks, threads>>>(
            input.data_ptr<float>(), 
            output.data_ptr<float>(), 
            num_elements
        );
    } else {
        // Fallback for very small tensors
        for (int64_t i = 0; i < num_elements; ++i) {
            output.data_ptr<float>()[i] = fmaxf(0.0f, input.data_ptr<float>()[i]);
        }
    }
}
"""

# C++ interface
cpp_source = r"""
#include <torch/extension.h>

void relu_forward_cuda_vec(const torch::Tensor& input, torch::Tensor& output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_relu", &relu_forward_cuda_vec, "Optimized Vectorized ReLU Forward");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_relu_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x):
    # Ensure input is on device
    if not x.is_cuda:
        x = x.to('cuda', non_blocking=True)
    
    # Pre-allocate output
    output = torch.empty_like(x)
    
    # Call the vectorized CUDA kernel
    fused_ext.fused_relu(x, output)
    
    return output

# Benchmark support
batch_size = 4096
dim = 393216

def get_init_inputs():
    return []

def get_inputs():
    x = torch.rand(batch_size, dim)
    return [x]
