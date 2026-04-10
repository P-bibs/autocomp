# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_233710/code_23.py
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

# Optimized CUDA kernel: Processes 8 floats per thread to increase Compute/Memory ratio
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void tanh_vec8_kernel(const float* __restrict__ x, float* __restrict__ out, size_t n) {
    // Each thread processes 8 elements (two float4)
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t idx = tid * 8;
    
    // Process main 8-element chunks
    if (idx + 7 < n) {
        float4 vec1 = reinterpret_cast<const float4*>(x)[tid * 2];
        float4 vec2 = reinterpret_cast<const float4*>(x)[tid * 2 + 1];
        
        vec1.x = tanhf(vec1.x); vec1.y = tanhf(vec1.y); 
        vec1.z = tanhf(vec1.z); vec1.w = tanhf(vec1.w);
        vec2.x = tanhf(vec2.x); vec2.y = tanhf(vec2.y); 
        vec2.z = tanhf(vec2.z); vec2.w = tanhf(vec2.w);
        
        reinterpret_cast<float4*>(out)[tid * 2] = vec1;
        reinterpret_cast<float4*>(out)[tid * 2 + 1] = vec2;
    } else {
        // Remainder handling
        for (size_t i = idx; i < n; ++i) {
            out[i] = tanhf(x[i]);
        }
    }
}

void fused_tanh_forward(torch::Tensor x, torch::Tensor out) {
    const size_t n = x.numel();
    const int threads = 256;
    // Divide work by 8 elements per thread
    const int blocks = (n / 8 + threads - 1) / threads;
    tanh_vec8_kernel<<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), n);
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_tanh_forward(torch::Tensor x, torch::Tensor out);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_tanh", &fused_tanh_forward, "Optimized Vectorized Tanh");
}
"""

# Compile extension
fused_tanh_module = load_inline(
    name='fused_tanh_optimized',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math', '--extra-device-vectorization'],
    with_cuda=True
)

def functional_model(x):
    # Ensure memory is contiguous to support float4 coalesced access
    # Performance is strictly bound by memory bandwidth provided contiguousness is met
    if not x.is_contiguous():
        x = x.contiguous()
        
    out = torch.empty_like(x)
    fused_tanh_module.fused_tanh(x, out)
    return out

def get_init_inputs():
    return []

def get_inputs():
    batch_size = 4096
    dim = 393216
    x = torch.rand(batch_size, dim, device='cuda', dtype=torch.float32)
    return [x]
