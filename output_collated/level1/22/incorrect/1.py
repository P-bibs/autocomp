# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_235928/code_11.py
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
# Inline CUDA kernel – optimized block size (1024) and 8 elements/thread
# ----------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// Vectorized kernel: each thread processes 8 floats (2x float4)
__global__ void fused_tanh_vec4_opt_kernel(const float* __restrict__ x,
                                            float*       __restrict__ out,
                                            const size_t n) {
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;

    // Process 8 elements via two float4 loads
    size_t idx0 = tid;
    if (idx0 + 3 < n) {
        float4 v0 = reinterpret_cast<const float4*>(x)[idx0 / 4];
        float4 r0;
        r0.x = tanhf(v0.x);
        r0.y = tanhf(v0.y);
        r0.z = tanhf(v0.z);
        r0.w = tanhf(v0.w);
        reinterpret_cast<float4*>(out)[idx0 / 4] = r0;
    } else {
        for (size_t i = idx0; i < n; ++i) out[i] = tanhf(x[i]);
        return;
    }

    size_t idx1 = tid + stride;
    if (idx1 + 3 < n) {
        float4 v1 = reinterpret_cast<const float4*>(x)[idx1 / 4];
        float4 r1;
        r1.x = tanhf(v1.x);
        r1.y = tanhf(v1.y);
        r1.z = tanhf(v1.z);
        r1.w = tanhf(v1.w);
        reinterpret_cast<float4*>(out)[idx1 / 4] = r1;
    } else {
        for (size_t i = idx1; i < n; ++i) out[i] = tanhf(x[i]);
    }
}

// Wrapper that launches the optimized kernel
void fused_tanh_forward(torch::Tensor x, torch::Tensor out) {
    const size_t n = x.numel();
    const int threads = 1024;  // Max for RTX 2080Ti
    const size_t per_block = static_cast<size_t>(threads) * 8;
    const int blocks = static_cast<int>((n + per_block - 1) / per_block);

    if (n >= 4) {
        fused_tanh_vec4_opt_kernel<<<blocks, threads>>>(
            x.data_ptr<float>(), out.data_ptr<float>(), n);
    } else {
        // Fallback for tiny inputs (< 4 elements)
        const int small_blocks = (n + threads - 1) / threads;
        fused_tanh_vec4_opt_kernel<<<small_blocks, threads>>>(
            x.data_ptr<float>(), out.data_ptr<float>(), n);
    }
}
"""

# ----------------------------------------------------------------------
# C++ binding (pybind11)
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>
void fused_tanh_forward(torch::Tensor x, torch::Tensor out);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_tanh", &fused_tanh_forward, "Optimized fused tanh forward");
}
"""

# ----------------------------------------------------------------------
# Compile the extension
# ----------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_tanh',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True,
)

# ----------------------------------------------------------------------
# Functional model used by the benchmark harness
# ----------------------------------------------------------------------
def functional_model(x: torch.Tensor) -> torch.Tensor:
    """
    Applies element-wise tanh using a custom CUDA kernel.
    The input is expected to be a contiguous float32 tensor on GPU.
    """
    if not x.is_contiguous():
        x = x.contiguous()
    out = torch.empty_like(x)
    fused_ext.fused_tanh(x, out)
    return out

# ----------------------------------------------------------------------
# Benchmark configuration
# ----------------------------------------------------------------------
batch_size = 4096
dim = 393216

def get_init_inputs():
    return []

def get_inputs():
    x = torch.rand(batch_size, dim, device='cuda', dtype=torch.float32)
    return [x]

# ----------------------------------------------------------------------
# End of file
# ----------------------------------------------------------------------
