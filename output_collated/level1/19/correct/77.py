# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_190701/code_28.py
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
# CUDA source – Optimised ReLU kernel with balanced occupancy
# -------------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// Vectorized ReLU using float4 to maximize bandwidth efficiency
// Each thread processes 8 floats total (2 float4 operations)
__global__ void relu_fused_kernel(const float* __restrict__ input,
                                  float* __restrict__ output,
                                  unsigned int n) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int base = tid * 8U;
    
    // Check if entire 8-element block is within bounds
    if (base + 7U < n) {
        // Load two float4 vectors
        float4 in1 = __ldg(reinterpret_cast<const float4*>(input + base));
        float4 in2 = __ldg(reinterpret_cast<const float4*>(input + base + 4));
        
        // Fused ReLU logic
        in1.x = fmaxf(in1.x, 0.0f); in1.y = fmaxf(in1.y, 0.0f);
        in1.z = fmaxf(in1.z, 0.0f); in1.w = fmaxf(in1.w, 0.0f);
        in2.x = fmaxf(in2.x, 0.0f); in2.y = fmaxf(in2.y, 0.0f);
        in2.z = fmaxf(in2.z, 0.0f); in2.w = fmaxf(in2.w, 0.0f);
        
        // Store results
        reinterpret_cast<float4*>(output + base)[0] = in1;
        reinterpret_cast<float4*>(output + base + 4)[0] = in2;
    } else {
        // Handle remainder for boundaries
        for (unsigned int i = base; i < n; ++i) {
            output[i] = fmaxf(input[i], 0.0f);
        }
    }
}

void fused_op_forward(torch::Tensor input, torch::Tensor output) {
    unsigned int n = static_cast<unsigned int>(input.numel());
    constexpr unsigned int threads = 256;
    // Each thread processes 8 elements
    unsigned int blocks = (n + (8 * threads) - 1U) / (8 * threads);

    relu_fused_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        n
    );
}
"""

# -------------------------------------------------------------------------
# C++ binding
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>
void fused_op_forward(torch::Tensor input, torch::Tensor output);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused ReLU kernel (optimized)");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x: torch.Tensor) -> torch.Tensor:
    """Applies ReLU on `x` using the optimized CUDA kernel."""
    output = torch.empty_like(x)
    fused_ext.fused_op(x, output)
    return output

# --- Benchmark harness compatibility ---
batch_size = 4096
dim = 393216

def get_init_inputs():
    return []

def get_inputs():
    x = torch.rand(batch_size, dim, device='cuda')
    return [x]
