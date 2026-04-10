# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_190701/code_0.py
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
# CUDA source – fused ReLU kernel (8-element vectorization + loop unrolling)
# -------------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------
// Optimized fused kernel with 8-element vectorization and loop unrolling
// ---------------------------------------------------------------------
__global__ void relu_fused_kernel(const float* __restrict__ input,
                                   float* __restrict__ output,
                                   size_t n) {
    // One thread per "slot" of 8 elements
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t base = tid * 8;                // First global index this thread handles
    if (base >= n) return;                // Out of range – early exit

    size_t remaining = n - base;          // How many valid elements are left

    // ----- bulk path: full float8 vector -----
    if (remaining >= 8) {
        // Load two float4 vectors to simulate 8-element processing
        float4 in_vec1 = __ldg(reinterpret_cast<const float4*>(input) + tid * 2);
        float4 in_vec2 = __ldg(reinterpret_cast<const float4*>(input) + tid * 2 + 1);
        
        float4 out_vec1, out_vec2;
        out_vec1.x = fmaxf(in_vec1.x, 0.0f);
        out_vec1.y = fmaxf(in_vec1.y, 0.0f);
        out_vec1.z = fmaxf(in_vec1.z, 0.0f);
        out_vec1.w = fmaxf(in_vec1.w, 0.0f);
        out_vec2.x = fmaxf(in_vec2.x, 0.0f);
        out_vec2.y = fmaxf(in_vec2.y, 0.0f);
        out_vec2.z = fmaxf(in_vec2.z, 0.0f);
        out_vec2.w = fmaxf(in_vec2.w, 0.0f);
        
        // Write the result as two float4 vectors (coalesced)
        reinterpret_cast<float4*>(output)[tid * 2] = out_vec1;
        reinterpret_cast<float4*>(output)[tid * 2 + 1] = out_vec2;
    }
    // ----- tail path: 1-7 elements, unrolled scalar accesses -----
    else {
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            if (i < remaining) {
                float v = __ldg(input + base + i);
                output[base + i] = fmaxf(v, 0.0f);
            }
        }
    }
}

// ---------------------------------------------------------------------
// Host driver – computes grid size and launches the fused kernel
// ---------------------------------------------------------------------
void fused_op_forward(torch::Tensor input, torch::Tensor output) {
    size_t n = input.numel();                     // Total number of floats
    constexpr int Threads = 1024;                 // Larger block for better occupancy

    // Number of "vector slots" = ceil(n / 8)
    size_t vector_cnt = (n + 7ULL) >> 3;
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
    output = torch.empty_like(x)          # Allocate output on the same device
    fused_ext.fused_op(x, output)         # Launch the fused kernel
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
