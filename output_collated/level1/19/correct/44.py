# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_185153/code_22.py
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

# Optimized CUDA Kernel:
# 1. Each thread processes 8 float4s (32 floats total) to hide instruction latency.
# 2. Uses #pragma unroll to eliminate loop overhead.
# 3. Aligns memory access to ensure 128-bit coalesced transactions.

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void relu_vec4_optimized_kernel(const float* __restrict__ input, float* __restrict__ output, size_t num_elements) {
    // Each thread processes 8 float4 (32 floats)
    const int VEC_PER_THREAD = 8;
    const int STRIDE = blockDim.x * gridDim.x * VEC_PER_THREAD;
    size_t idx = (size_t)(blockIdx.x * blockDim.x + threadIdx.x) * VEC_PER_THREAD;

    #pragma unroll
    for (int i = 0; i < VEC_PER_THREAD; ++i) {
        size_t current_vec_idx = idx + i;
        if (current_vec_idx * 4 < num_elements) {
            float4 val = reinterpret_cast<const float4*>(input)[current_vec_idx];
            
            val.x = fmaxf(0.0f, val.x);
            val.y = fmaxf(0.0f, val.y);
            val.z = fmaxf(0.0f, val.z);
            val.w = fmaxf(0.0f, val.w);
            
            reinterpret_cast<float4*>(output)[current_vec_idx] = val;
        }
    }
}

void relu_launch(torch::Tensor input, torch::Tensor output) {
    size_t num_elements = input.numel();
    size_t num_vec4 = (num_elements + 3) / 4;
    
    // Configuration for RTX 2080Ti (Turing architecture)
    const int threads = 256;
    const int vec_per_thread = 8;
    const int blocks = (num_vec4 + (threads * vec_per_thread) - 1) / (threads * vec_per_thread);
    
    relu_vec4_optimized_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        num_elements
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void relu_launch(torch::Tensor input, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("relu_vec4", &relu_launch, "Optimized vectorized ReLU kernel");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_relu_opt',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x):
    """
    High-performance ReLU using vectorized CUDA kernels.
    """
    output = torch.empty_like(x)
    fused_ext.relu_vec4(x, output)
    return output
