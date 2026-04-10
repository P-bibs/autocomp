# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_190701/code_8.py
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
# CUDA source – optimized ReLU kernel with increased work per thread
# -------------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// Optimized kernel: each thread processes 8 elements (2 x float4)
// This increases arithmetic intensity and improves memory bandwidth utilization
__global__ void relu_fused_kernel(const float* __restrict__ input,
                                   float* __restrict__ output,
                                   size_t n) {
    // one thread per "batch" of 8 elements
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t base = tid * 8;                // first global index this thread handles
    
    if (base >= n) return;                // out of range – early exit

    size_t remaining = n - base;          // how many valid elements are left

    // Process first batch of 4 elements (if available)
    if (remaining >= 4) {
        float4 in_vec = __ldg(reinterpret_cast<const float4*>(input) + tid * 2);
        float4 out_vec;
        out_vec.x = fmaxf(in_vec.x, 0.0f);
        out_vec.y = fmaxf(in_vec.y, 0.0f);
        out_vec.z = fmaxf(in_vec.z, 0.0f);
        out_vec.w = fmaxf(in_vec.w, 0.0f);
        reinterpret_cast<float4*>(output)[tid * 2] = out_vec;
        
        // Process second batch of 4 elements (if available)
        remaining -= 4;
        if (remaining >= 4) {
            in_vec = __ldg(reinterpret_cast<const float4*>(input) + tid * 2 + 1);
            out_vec.x = fmaxf(in_vec.x, 0.0f);
            out_vec.y = fmaxf(in_vec.y, 0.0f);
            out_vec.z = fmaxf(in_vec.z, 0.0f);
            out_vec.w = fmaxf(in_vec.w, 0.0f);
            reinterpret_cast<float4*>(output)[tid * 2 + 1] = out_vec;
        }
        else if (remaining > 0) {
            // Handle tail (1-3 elements)
            const size_t base_offset = base + 4;
            if (remaining > 0) {
                float v = __ldg(input + base_offset);
                output[base_offset] = fmaxf(v, 0.0f);
            }
            if (remaining > 1) {
                float v = __ldg(input + base_offset + 1);
                output[base_offset + 1] = fmaxf(v, 0.0f);
            }
            if (remaining > 2) {
                float v = __ldg(input + base_offset + 2);
                output[base_offset + 2] = fmaxf(v, 0.0f);
            }
        }
    }
    else {
        // Handle partial first batch (1-3 elements)
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

// Host driver – computes grid size and launches the optimized kernel
void fused_op_forward(torch::Tensor input, torch::Tensor output) {
    size_t n = input.numel();
    constexpr int Threads = 512;  // Reduced thread count for better occupancy
    
    // Number of "batches of 8" = ceil(n / 8)
    size_t batch_cnt = (n + 7ULL) >> 3;
    int blocks = static_cast<int>((batch_cnt + Threads - 1) / Threads);

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
    m.def("fused_op", &fused_op_forward, "Optimized fused ReLU kernel with batched work per thread");
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
# Functional wrapper
# -------------------------------------------------------------------------
def functional_model(x: torch.Tensor) -> torch.Tensor:
    """Applies ReLU on `x` using the optimized batched kernel."""
    output = torch.empty_like(x)
    fused_ext.fused_op(x, output)
    return output

# -------------------------------------------------------------------------
# Benchmark definitions
# -------------------------------------------------------------------------
batch_size = 4096
dim = 393216

def get_init_inputs():
    return []

def get_inputs():
    x = torch.rand(batch_size, dim, device='cuda')
    return [x]
