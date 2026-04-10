# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_012911/code_31.py
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
# CUDA Kernel: Vectorized float4 load/store with loop unrolling
# -------------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector_types.h>

// Each thread processes 8 float4 vectors (8 * 4 = 32 elements)
constexpr int ELEMENTS_PER_THREAD = 8;

__global__ void tanh_opt_kernel(const float* __restrict__ x,
                                 float* __restrict__ out,
                                 size_t n) {
    size_t start = (blockIdx.x * blockDim.x + threadIdx.x) * (ELEMENTS_PER_THREAD * 4);
    
    // Process main body with unrolled loop
    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
        size_t idx = start + i * 4;
        if (idx + 3 < n) {
            float4 vec_x = reinterpret_cast<const float4*>(x + idx)[0];
            float4 vec_out;
            vec_out.x = tanhf(vec_x.x);
            vec_out.y = tanhf(vec_x.y);
            vec_out.z = tanhf(vec_x.z);
            vec_out.w = tanhf(vec_x.w);
            reinterpret_cast<float4*>(out + idx)[0] = vec_out;
        } else {
            // Handle remainder
            for (int k = 0; k < 4; ++k) {
                if (idx + k < n) {
                    out[idx + k] = tanhf(x[idx + k]);
                }
            }
        }
    }
}

void fused_tanh_forward(torch::Tensor x, torch::Tensor out) {
    const size_t n = x.numel();
    const int threads = 256;
    const size_t elements_per_block = (size_t)threads * ELEMENTS_PER_THREAD * 4;
    const int blocks = (int)((n + elements_per_block - 1) / elements_per_block);
    
    tanh_opt_kernel<<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), n);
}
"""

# -------------------------------------------------------------------------
# C++ Bindings
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>
void fused_tanh_forward(torch::Tensor x, torch::Tensor out);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_tanh", &fused_tanh_forward, "Optimized Vectorized Tanh forward");
}
"""

# Compile
fused_ext = load_inline(
    name='fused_tanh_opt',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x: torch.Tensor) -> torch.Tensor:
    """
    Optimized tanh implementation using a custom CUDA kernel.
    Input is force-converted to contiguous to ensure float4 alignment.
    """
    if not x.is_contiguous():
        x = x.contiguous()
    out = torch.empty_like(x)
    fused_ext.fused_tanh(x, out)
    return out
