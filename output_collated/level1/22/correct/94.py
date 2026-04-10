# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_004356/code_7.py
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

// Process 8 floats per thread (two float4 loads) to maximize ILP and reduce launch overhead
__global__ void tanh_optimized_kernel(const float* __restrict__ x, float* __restrict__ out, size_t n) {
    size_t vec_idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    
    if (vec_idx + 7 < n) {
        float4 v1 = reinterpret_cast<const float4*>(x + vec_idx)[0];
        float4 v2 = reinterpret_cast<const float4*>(x + vec_idx + 4)[0];
        
        v1.x = tanhf(v1.x); v1.y = tanhf(v1.y); v1.z = tanhf(v1.z); v1.w = tanhf(v1.w);
        v2.x = tanhf(v2.x); v2.y = tanhf(v2.y); v2.z = tanhf(v2.z); v2.w = tanhf(v2.w);
        
        reinterpret_cast<float4*>(out + vec_idx)[0] = v1;
        reinterpret_cast<float4*>(out + vec_idx + 4)[0] = v2;
    } else {
        // Scalar fallback for remainder
        for (size_t i = vec_idx; i < n; ++i) {
            out[i] = tanhf(x[i]);
        }
    }
}

void fused_tanh_forward(torch::Tensor x, torch::Tensor out) {
    const size_t n = x.numel();
    const int threads = 256;
    // Process 8 elements per thread
    const int blocks = (n + (threads * 8) - 1) / (threads * 8);
    tanh_optimized_kernel<<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), n);
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_tanh_forward(torch::Tensor x, torch::Tensor out);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_tanh", &fused_tanh_forward, "Optimized Tanh forward");
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
    # Ensure input is contiguous for float4 vectorization
    if not x.is_contiguous():
        x = x.contiguous()
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
