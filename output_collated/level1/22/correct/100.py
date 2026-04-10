# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_004356/code_15.py
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
# CUDA source – vectorised tanh kernel (8 elements per thread)
# -------------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// Each thread processes 8 floats (two float4 vectors)
__global__ void tanh_vec8_kernel(const float* __restrict__ x,
                                 float*       __restrict__ out,
                                 const size_t n) {
    // index of the first element handled by this thread
    const size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;

    if (idx + 7 < n) {                     // fully‑occupied 8‑element tile
        // ---- first float4 ----
        float4 vec_x0 = reinterpret_cast<const float4*>(x)[idx >> 2];
        float4 vec_o0;
        vec_o0.x = tanhf(vec_x0.x);
        vec_o0.y = tanhf(vec_x0.y);
        vec_o0.z = tanhf(vec_x0.z);
        vec_o0.w = tanhf(vec_x0.w);
        reinterpret_cast<float4*>(out)[idx >> 2] = vec_o0;

        // ---- second float4 ----
        const size_t idx2 = idx + 4;
        float4 vec_x1 = reinterpret_cast<const float4*>(x)[idx2 >> 2];
        float4 vec_o1;
        vec_o1.x = tanhf(vec_x1.x);
        vec_o1.y = tanhf(vec_x1.y);
        vec_o1.z = tanhf(vec_x1.z);
        vec_o1.w = tanhf(vec_x1.w);
        reinterpret_cast<float4*>(out)[idx2 >> 2] = vec_o1;
    } else {                                // tail (0‑7 elements)
        for (size_t i = idx; i < n; ++i) {
            out[i] = tanhf(x[i]);
        }
    }
}

// Host wrapper that chooses a grid large enough to cover all elements
void fused_tanh_forward(torch::Tensor x, torch::Tensor out) {
    const size_t n   = x.numel();
    const int    threads = 1024;                 // larger block → better occupancy
    const int    blocks  = (n + threads * 8 - 1) / (threads * 8);
    tanh_vec8_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), n);
}
"""

# -------------------------------------------------------------------------
# C++ binding (PYBIND11)
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>
void fused_tanh_forward(torch::Tensor x, torch::Tensor out);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_tanh", &fused_tanh_forward, "Vectorized tanh forward");
}
"""

# -------------------------------------------------------------------------
# Build the inline extension
# -------------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_tanh',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# -------------------------------------------------------------------------
# Functional model used by the evaluator
# -------------------------------------------------------------------------
def functional_model(x: torch.Tensor) -> torch.Tensor:
    """Apply the custom fused tanh to the input tensor."""
    # Ensure the tensor is contiguous – required for safe reinterpret_cast
    if not x.is_contiguous():
        x = x.contiguous()
    out = torch.empty_like(x)
    fused_ext.fused_tanh(x, out)
    return out

# -------------------------------------------------------------------------
# Benchmark‑setup helpers (unchanged)
# -------------------------------------------------------------------------
batch_size = 4096
dim = 393216

def get_init_inputs():
    return []

def get_inputs():
    # Random input on the GPU, float32, row‑major
    x = torch.rand(batch_size, dim, device='cuda', dtype=torch.float32)
    return [x]
