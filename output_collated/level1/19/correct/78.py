# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_190701/code_29.py
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

# --- Optimized CUDA Implementation ---
# Optimization Strategies:
# 1. Grid-Stride Loop: Each thread processes a sequence of float4 data, 
#    amortizing launch overhead and ensuring optimal latency on RTX 2080Ti.
# 2. Vectorized Memory Access: Loading/Storing float4 ensures coalesced
#    accesses, maximizing the 616 GB/s memory bandwidth of the 2080Ti.
# 3. Static Launch Configuration: Avoids dynamic overhead; 256 threads 
#    provide high occupancy, 8192 blocks keeps the SMs well-fed.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_op_forward_kernel(const float* __restrict__ input,
                                        float*       __restrict__ output,
                                        size_t n) {
    size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    const size_t stride = blockDim.x * gridDim.x * 4;

    // Grid-stride loop: each thread iterates over chunks of the tensor
    while (idx + 3 < n) {
        float4 in_vec = reinterpret_cast<const float4*>(input)[idx / 4];
        float4 out_vec;
        
        // ReLU fmaxf map
        out_vec.x = fmaxf(in_vec.x, 0.0f);
        out_vec.y = fmaxf(in_vec.y, 0.0f);
        out_vec.z = fmaxf(in_vec.z, 0.0f);
        out_vec.w = fmaxf(in_vec.w, 0.0f);
        
        reinterpret_cast<float4*>(output)[idx / 4] = out_vec;
        idx += stride;
    }

    // Handle remaining elements if n is not a multiple of 4
    for (size_t i = 0; i < 4 && idx + i < n; ++i) {
        output[idx + i] = fmaxf(input[idx + i], 0.0f);
    }
}

void fused_op_forward(torch::Tensor input, torch::Tensor output) {
    size_t n = input.numel();
    const int threads = 256;
    const int blocks = 8192; // Sufficient to keep 2080Ti occupancy high
    
    fused_op_forward_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        n
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(torch::Tensor input, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Grid-stride vectorized ReLU");
}
"""

# Build the extension inline
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x):
    """
    Applies ReLU via a custom high-performance CUDA kernel.
    """
    # Create output tensor with existing metadata
    output = torch.empty_like(x)
    # Invoke the optimized C++/CUDA function
    fused_ext.fused_op(x, output)
    return output

# --- Evaluation setup ---
batch_size = 4096
dim = 393216

def get_init_inputs():
    return []

def get_inputs():
    x = torch.rand(batch_size, dim, device='cuda')
    return [x]
