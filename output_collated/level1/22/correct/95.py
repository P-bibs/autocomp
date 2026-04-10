# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_004356/code_11.py
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

# Optimized CUDA kernel with separated main and remainder paths
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector_types.h>

__global__ void tanh_vec4_kernel(const float* __restrict__ x, float* __restrict__ out, const size_t n_full) {
    // Each thread processes exactly 4 elements via float4
    // No divergence: all threads follow the fast path
    const size_t idx = (static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 4;
    if (idx < n_full) {
        const float4 vec_x = reinterpret_cast<const float4*>(x)[idx / 4];
        float4 vec_out;
        vec_out.x = tanhf(vec_x.x);
        vec_out.y = tanhf(vec_x.y);
        vec_out.z = tanhf(vec_x.z);
        vec_out.w = tanhf(vec_x.w);
        reinterpret_cast<float4*>(out)[idx / 4] = vec_out;
    }
}

__global__ void tanh_remainder_kernel(const float* __restrict__ x, float* __restrict__ out, 
                                      const size_t offset, const size_t remainder) {
    // Dedicated kernel for remaining elements (0-3)
    // Only a few threads are active, no warp divergence impact
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < remainder) {
        out[offset + idx] = tanhf(x[offset + idx]);
    }
}

void fused_tanh_forward(torch::Tensor x, torch::Tensor out) {
    const size_t n = x.numel();
    const size_t n_full = (n / 4) * 4;  // Aligned portion
    const size_t remainder = n % 4;     // Remaining elements
    
    const int threads = 256;
    
    // Main kernel: process all 4-element aligned chunks
    if (n_full > 0) {
        const int blocks_main = (static_cast<int>(n_full) / 4 + threads - 1) / threads;
        tanh_vec4_kernel<<<blocks_main, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), n_full);
    }
    
    // Remainder kernel: only if there are remaining elements
    if (remainder > 0) {
        const int blocks_rem = 1;  // Minimal overhead for just a few elements
        tanh_remainder_kernel<<<blocks_rem, threads>>>(
            x.data_ptr<float>(), out.data_ptr<float>(), n_full, remainder
        );
    }
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_tanh_forward(torch::Tensor x, torch::Tensor out);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_tanh", &fused_tanh_forward, "Optimized vectorized Tanh forward without warp divergence");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_tanh',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x):
    out = torch.empty_like(x)
    fused_ext.fused_tanh(x, out)
    return out

# Configuration
batch_size = 4096
dim = 393216

def get_init_inputs():
    return []

def get_inputs():
    # Ensure data is contiguous for valid float4 casts
    x = torch.rand(batch_size, dim, device='cuda', dtype=torch.float32)
    return [x]
