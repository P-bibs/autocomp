# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_233710/code_31.py
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

# -------------------------------------------------------------------------
# Optimized CUDA kernel
# -------------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector_types.h>
#include <math.h>

__global__ void fused_tanh_kernel(const float* __restrict__ x,
                                   float*       __restrict__ out,
                                   const size_t n)
{
    // Offset by thread and block indices
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;
    
    // Each thread processes 4 elements (1 float4) per iteration
    // Unrolling helps the compiler pipeline the memory instructions
    for (size_t i = tid * 4; i < (n & ~3); i += stride * 4) {
        // float4 load/store for memory alignment/Coalescing
        float4 vec_x = reinterpret_cast<const float4*>(x)[i >> 2];
        float4 vec_out;
        
        // __builtin_tanhf usually maps to the hardware's fast_math reciprocal
        vec_out.x = tanhf(vec_x.x);
        vec_out.y = tanhf(vec_x.y);
        vec_out.z = tanhf(vec_x.z);
        vec_out.w = tanhf(vec_x.w);
        
        reinterpret_cast<float4*>(out)[i >> 2] = vec_out;
    }

    // Scalar cleanup loop for remaining elements
    // Typically 0-3 elements per thread path
    const size_t start_remainder = (n & ~3);
    for (size_t i = start_remainder + tid; i < n; i += stride) {
        out[i] = tanhf(x[i]);
    }
}

void fused_tanh_forward(torch::Tensor x, torch::Tensor out) {
    const size_t n = x.numel();
    const int threads = 256;
    // Heuristic: Ensure enough blocks to saturate SMs, but not so many to create thrashing
    // 128 blocks is a standard balance for 68 SMs on 2080Ti
    const int blocks = 512; 
    
    fused_tanh_kernel<<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), n);
}
"""

# -------------------------------------------------------------------------
# C++ binding
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>
void fused_tanh_forward(torch::Tensor x, torch::Tensor out);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_tanh", &fused_tanh_forward, "Vectorized Tanh forward (grid-stride)");
}
"""

# -------------------------------------------------------------------------
# Compilation
# -------------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_tanh',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math', '-arch=sm_75'],
    with_cuda=True
)

def functional_model(x: torch.Tensor) -> torch.Tensor:
    # Ensure optimal memory layout for coalesced access
    if not x.is_contiguous():
        x = x.contiguous()
    out = torch.empty_like(x)
    fused_ext.fused_tanh(x, out)
    return out

def get_init_inputs():
    return []

def get_inputs():
    x = torch.rand(4096, 393216, device='cuda', dtype=torch.float32)
    return [x]
