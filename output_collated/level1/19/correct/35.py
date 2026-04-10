# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_185153/code_12.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = []
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = []
REQUIRED_FLAT_STATE_NAMES = []


class ModelNew(nn.Module):
    """
    Simple model that performs a ReLU activation.
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
# CUDA source – fused ReLU kernel (vector + remainder) + host launcher
# -------------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------
// Fused kernel: each thread processes up to 4 elements (float4).
// If less than 4 elements remain, we fall back to scalar loads/stores.
// ---------------------------------------------------------------------
__global__ void relu_fused_kernel(const float* __restrict__ input,
                                   float* __restrict__ output,
                                   size_t n) {
    // one thread per "slot" of 4 elements
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t base = tid * 4;                // first global index this thread handles
    if (base >= n) return;                // out of range – early exit

    size_t remaining = n - base;          // how many valid elements are left

    // ----- bulk path: full float4 vector -----
    if (remaining >= 4) {
        // read a float4 through the read-only data cache
        float4 in_vec = __ldg(reinterpret_cast<const float4*>(input) + tid);
        float4 out_vec;
        out_vec.x = fmaxf(in_vec.x, 0.0f);
        out_vec.y = fmaxf(in_vec.y, 0.0f);
        out_vec.z = fmaxf(in_vec.z, 0.0f);
        out_vec.w = fmaxf(in_vec.w, 0.0f);
        // write the result as a float4 (coalesced)
        reinterpret_cast<float4*>(output)[tid] = out_vec;
    }
    // ----- tail path: 1-3 elements, scalar accesses -----
    else {
        if (remaining > 0) {
            float v = __ldg(input + base);
            output[base] = fmaxf(v, 0.0f);
        }
        if (remaining > 1) {
            float v = __ldg(input + base + 1);
            output[base + 1] = fmaxf(v, 0.0f);
        }
        if (remaining > 2) {
            float v = __ldg(input + base + 2);
            output[base + 2] = fmaxf(v, 0.0f);
        }
    }
}

// ---------------------------------------------------------------------
// Host driver – computes grid size and launches the fused kernel
// ---------------------------------------------------------------------
void fused_op_forward(torch::Tensor input, torch::Tensor output) {
    size_t n = input.numel();                     // total number of floats
    constexpr int Threads = 1024;                 // larger block for better occupancy

    // number of "vector slots" = ceil(n / 4)
    size_t vector_cnt = (n + 3ULL) >> 2;
    int blocks = static_cast<int>((vector_cnt + Threads - 1) / Threads);

    relu_fused_kernel<<<blocks, Threads>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), n);
}
"""

# -------------------------------------------------------------------------
# C++ binding (pybind11)
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>
void fused_op_forward(torch::Tensor input, torch::Tensor output);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused ReLU kernel");
}
"""

# -------------------------------------------------------------------------
# Build the inline extension
# -------------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# -------------------------------------------------------------------------
# Functional wrapper used by the benchmark harness
# -------------------------------------------------------------------------
def functional_model(x: torch.Tensor) -> torch.Tensor:
    """Applies ReLU on `x` using the fused single-kernel implementation."""
    output = torch.empty_like(x)          # allocate output on the same device
    fused_ext.fused_op(x, output)         # launch the fused kernel
    return output

# -------------------------------------------------------------------------
# Benchmark-related definitions (kept for reference)
# -------------------------------------------------------------------------
batch_size = 4096
dim = 393216

def get_init_inputs():
    return []

def get_inputs():
    x = torch.rand(batch_size, dim, device='cuda')
    return [x]
