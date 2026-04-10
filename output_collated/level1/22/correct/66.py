# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_001958/code_9.py
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

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector_types.h>

__global__ void tanh_optimized_kernel(const float* __restrict__ x, float* __restrict__ out, size_t n) {
    // Each thread processes 8 floats (2 float4 vectors) sequentially
    // This reduces thread count needed and improves occupancy
    size_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    size_t elements_per_thread = 8;
    size_t base_idx = thread_id * elements_per_thread;
    
    // Main vectorized path: process 8 elements per thread
    if (base_idx + 7 < n) {
        // Process two float4 vectors
        float4 vec_x1 = reinterpret_cast<const float4*>(x)[base_idx / 4];
        float4 vec_x2 = reinterpret_cast<const float4*>(x)[(base_idx + 4) / 4];
        
        float4 vec_out1, vec_out2;
        vec_out1.x = tanhf(vec_x1.x);
        vec_out1.y = tanhf(vec_x1.y);
        vec_out1.z = tanhf(vec_x1.z);
        vec_out1.w = tanhf(vec_x1.w);
        
        vec_out2.x = tanhf(vec_x2.x);
        vec_out2.y = tanhf(vec_x2.y);
        vec_out2.z = tanhf(vec_x2.z);
        vec_out2.w = tanhf(vec_x2.w);
        
        reinterpret_cast<float4*>(out)[base_idx / 4] = vec_out1;
        reinterpret_cast<float4*>(out)[(base_idx + 4) / 4] = vec_out2;
    }
    // Handle partial block: process remaining elements (0-7 elements)
    else if (base_idx < n) {
        size_t remaining = n - base_idx;
        for (size_t i = 0; i < remaining; ++i) {
            out[base_idx + i] = tanhf(x[base_idx + i]);
        }
    }
}

void fused_tanh_forward(torch::Tensor x, torch::Tensor out) {
    const size_t n = x.numel();
    const int threads = 512;  // Increased from 256 to improve occupancy
    const int elements_per_thread = 8;
    
    // Calculate blocks needed: each thread handles 8 elements
    const int blocks = (n + threads * elements_per_thread - 1) / (threads * elements_per_thread);
    
    tanh_optimized_kernel<<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), n);
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_tanh_forward(torch::Tensor x, torch::Tensor out);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_tanh", &fused_tanh_forward, "High-occupancy vectorized tanh forward");
}
"""

# Compile with optimization flags
fused_ext = load_inline(
    name='fused_tanh_opt',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x):
    """
    High-occupancy optimized tanh operation.
    Ensures input is contiguous for safe reinterpret_cast to float4.
    """
    if not x.is_contiguous():
        x = x.contiguous()
    out = torch.empty_like(x)
    fused_ext.fused_tanh(x, out)
    return out

def get_inputs():
    batch_size = 4096
    dim = 393216
    x = torch.rand(batch_size, dim, device='cuda', dtype=torch.float32)
    return [x]
