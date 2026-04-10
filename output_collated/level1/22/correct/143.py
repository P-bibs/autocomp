# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_010905/code_29.py
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

# Optimized CUDA kernel: 
# 1. Using Grid-Stride Loop pattern for better scaling and load balancing.
# 2. Vectorized 128-bit loads (float4) for memory bandwidth saturation.
# 3. Fast-math enabled for hardware-level approximation of transcendental functions.

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void tanh_optimized_kernel(const float* __restrict__ x, float* __restrict__ out, size_t n) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    // We process 4 elements per iteration using float4 to ensure 128-bit memory alignment
    // This assumes n is divisible by 4 (handled by padding/contiguous check in Python)
    for (size_t i = tid * 4; i < n; i += stride * 4) {
        float4 vec_x = reinterpret_cast<const float4*>(x)[i / 4];
        float4 vec_out;
        vec_out.x = tanhf(vec_x.x);
        vec_out.y = tanhf(vec_x.y);
        vec_out.z = tanhf(vec_x.z);
        vec_out.w = tanhf(vec_x.w);
        reinterpret_cast<float4*>(out)[i / 4] = vec_out;
    }
}

void fused_tanh_forward(torch::Tensor x, torch::Tensor out) {
    const size_t n = x.numel();
    const int threads = 256;
    // Determine grid size: aim for ~65536 max blocks for Turing
    const int blocks = std::min((int)((n / 4 + threads - 1) / threads), 4096);
    
    tanh_optimized_kernel<<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), n);
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_tanh_forward(torch::Tensor x, torch::Tensor out);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_tanh", &fused_tanh_forward, "Optimized Vectorized Tanh forward");
}
"""

fused_ext = load_inline(
    name='fused_tanh_opt',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x):
    """
    Optimized tanh operation.
    Ensures input is contiguous and padded for float4 alignment.
    """
    if not x.is_contiguous():
        x = x.contiguous()
    
    n = x.numel()
    # Ensure n is divisible by 4 for float4 reinterpret_cast safety
    if n % 4 != 0:
        padding = 4 - (n % 4)
        x_flat = x.view(-1)
        x_padded = torch.nn.functional.pad(x_flat, (0, padding))
        out_padded = torch.empty_like(x_padded)
        fused_ext.fused_tanh(x_padded, out_padded)
        return out_padded.narrow(0, 0, n).view_as(x)
    else:
        out = torch.empty_like(x)
        fused_ext.fused_tanh(x, out)
        return out
