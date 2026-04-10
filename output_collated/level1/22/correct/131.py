# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_010905/code_15.py
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
# CUDA kernel – 8 floats per thread (two float4 vectors)
# ----------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector_types.h>

constexpr int BLOCK_SIZE = 256;
constexpr int ELEMENTS_PER_THREAD = 8;   // 8 floats = 2 * float4

__global__ void tanh_vec8_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    size_t n)
{
    // index of the first element this thread will process
    size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * ELEMENTS_PER_THREAD;
    if (idx >= n) return;                     // out of range guard

    // Full‑vector path when we have at least 8 elements left
    if (idx + ELEMENTS_PER_THREAD <= n) {
        const float4* x4 = reinterpret_cast<const float4*>(x);
        float4*       o4 = reinterpret_cast<float4*>(out);

        // Load two float4 vectors (8 floats)
        float4 v0 = x4[idx >> 2];          // idx/4
        float4 v1 = x4[(idx + 4) >> 2];    // (idx+4)/4

        // Compute tanh for each component
        float4 r0, r1;
        r0.x = tanhf(v0.x);
        r0.y = tanhf(v0.y);
        r0.z = tanhf(v0.z);
        r0.w = tanhf(v0.w);

        r1.x = tanhf(v1.x);
        r1.y = tanhf(v1.y);
        r1.z = tanhf(v1.z);
        r1.w = tanhf(v1.w);

        // Store results
        o4[idx >> 2]       = r0;
        o4[(idx + 4) >> 2] = r1;
    }
    else {
        // Remainder – at most 7 elements, processed element‑wise
        for (size_t i = idx; i < n; ++i) {
            out[i] = tanhf(x[i]);
        }
    }
}

// Host‑side launcher
void fused_tanh_forward(torch::Tensor x, torch::Tensor out) {
    const size_t n = x.numel();
    const int threads = BLOCK_SIZE;
    const int vec_width = ELEMENTS_PER_THREAD;
    const int blocks = static_cast<int>((n + threads * vec_width - 1) / (threads * vec_width));

    tanh_vec8_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        n);
}
"""

# ----------------------------------------------------------------------
# C++ binding (pybind11)
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>
void fused_tanh_forward(torch::Tensor x, torch::Tensor out);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_tanh", &fused_tanh_forward, "Vectorized tanh forward");
}
"""

# ----------------------------------------------------------------------
# Build the inline extension
# ----------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_tanh',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# ----------------------------------------------------------------------
# Functional model – the only entry point used during evaluation
# ----------------------------------------------------------------------
def functional_model(x: torch.Tensor) -> torch.Tensor:
    """
    Applies element‑wise tanh to the input tensor using the custom
    vectorized CUDA kernel.
    """
    # Ensure the input is contiguous (guarantees proper alignment for float4 loads)
    if not x.is_contiguous():
        x = x.contiguous()
    out = torch.empty_like(x)
    fused_ext.fused_tanh(x, out)
    return out


# ----------------------------------------------------------------------
# Helper functions for the evaluation harness (not required for the kernel)
# ----------------------------------------------------------------------
def get_init_inputs():
    return []


def get_inputs():
    batch_size = 4096
    dim = 393216
    # Random input on GPU, float32, contiguous
    x = torch.rand(batch_size, dim, device='cuda', dtype=torch.float32)
    return [x]
