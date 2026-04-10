# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_010905/code_31.py
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
# CUDA kernel – Vectorized 8 floats per thread (two float4 vectors)
# Maximizes throughput and kernel-launch efficiency.
# ----------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector_types.h>

// Process 8 floats per thread using two float4 loads
#define ELEMENTS_PER_THREAD 8

__global__ void tanh_vec8_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    size_t n)
{
    size_t idx = (size_t)(blockIdx.x * blockDim.x + threadIdx.x) * ELEMENTS_PER_THREAD;
    
    // Total float4 vectors to process
    size_t n_v4 = n / 4;
    size_t vec_idx = idx / 4;

    // Full-vector path
    if (idx + 7 < n) {
        const float4* x4 = reinterpret_cast<const float4*>(x);
        float4*       o4 = reinterpret_cast<float4*>(out);

        float4 v0 = x4[vec_idx];
        float4 v1 = x4[vec_idx + 1];

        v0.x = tanhf(v0.x);
        v0.y = tanhf(v0.y);
        v0.z = tanhf(v0.z);
        v0.w = tanhf(v0.w);

        v1.x = tanhf(v1.x);
        v1.y = tanhf(v1.y);
        v1.z = tanhf(v1.z);
        v1.w = tanhf(v1.w);

        o4[vec_idx]     = v0;
        o4[vec_idx + 1] = v1;
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
    // Each block processes 256 * 8 = 2048 elements
    const int blocks = (n + (threads * ELEMENTS_PER_THREAD) - 1) / (threads * ELEMENTS_PER_THREAD);
    
    tanh_vec8_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        n);
}
"""

# ----------------------------------------------------------------------
# C++ binding
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>
void fused_tanh_forward(torch::Tensor x, torch::Tensor out);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_tanh", &fused_tanh_forward, "Vectorized Tanh forward (8 floats/thread)");
}
"""

# ----------------------------------------------------------------------
# Build extension
# ----------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_tanh_opt',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x: torch.Tensor) -> torch.Tensor:
    """
    Applies element-wise tanh to the input tensor.
    The input is guaranteed to be contiguous by the performance harness 
    or ensured here for alignment.
    """
    if not x.is_contiguous():
        x = x.contiguous()
    out = torch.empty_like(x)
    fused_ext.fused_tanh(x, out)
    return out

def get_init_inputs():
    return []

def get_inputs():
    batch_size = 4096
    dim = 393216
    x = torch.rand(batch_size, dim, device='cuda', dtype=torch.float32)
    return [x]
