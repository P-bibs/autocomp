# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_235928/code_23.py
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

# ----------------------------------------------------------------------
# Optimized CUDA Kernel
# We use a large blocks-per-grid configuration and a higher registers-per-thread
# count by processing 8 elements per thread using float4 vector instructions.
# This maximizes memory bandwidth utilization on the RTX 2080Ti.
# ----------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector_types.h>

__global__ void fused_tanh_vec8_kernel(const float* __restrict__ x, float* __restrict__ out, size_t n) {
    // Each thread processes 8 elements (2x float4)
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    size_t vec_stride = stride / 4;
    size_t vec_idx = tid;

    // Process using float4 to ensure 128-bit memory transactions (coalesced)
    // We unroll the loop to help the compiler pipeline instructions
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        size_t current_vec = vec_idx + i * vec_stride;
        if ((current_vec * 4) + 3 < n) {
            float4 v = reinterpret_cast<const float4*>(x)[current_vec];
            float4 res;
            res.x = tanhf(v.x);
            res.y = tanhf(v.y);
            res.z = tanhf(v.z);
            res.w = tanhf(v.w);
            reinterpret_cast<float4*>(out)[current_vec] = res;
        } else {
            // Remainder handling
            for (int k = 0; k < 4; ++k) {
                size_t abs_idx = current_vec * 4 + k;
                if (abs_idx < n) {
                    out[abs_idx] = tanhf(x[abs_idx]);
                }
            }
        }
    }
}

void fused_tanh_forward(torch::Tensor x, torch::Tensor out) {
    const size_t n = x.numel();
    // 1024 threads is optimal for RTX 2080Ti occupancy
    const int threads = 1024;
    // Each thread processes 8 floats. Total elements per block = 8192
    const int elements_per_thread = 8;
    const int blocks = (n + (threads * elements_per_thread) - 1) / (threads * elements_per_thread);
    
    fused_tanh_vec8_kernel<<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), n);
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_tanh_forward(torch::Tensor x, torch::Tensor out);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_tanh", &fused_tanh_forward, "Vectorized Tanh forward");
}
"""

# Compile the extension with optimized flags
fused_ext = load_inline(
    name='fused_tanh',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math', '-arch=sm_75'],
    with_cuda=True
)

def functional_model(x: torch.Tensor) -> torch.Tensor:
    """
    Optimized tanh implementation using a custom CUDA kernel.
    Ensures input is contiguous to abide by vectorized alignment requirements.
    """
    if not x.is_contiguous():
        x = x.contiguous()
    out = torch.empty_like(x)
    fused_ext.fused_tanh(x, out)
    return out

# Configuration for consistency
batch_size = 4096
dim = 393216

def get_init_inputs():
    return []

def get_inputs():
    x = torch.rand(batch_size, dim, device='cuda', dtype=torch.float32)
    return [x]
