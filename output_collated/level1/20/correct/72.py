# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_211408/code_31.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['negative_slope']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['negative_slope']
REQUIRED_FLAT_STATE_NAMES = []


class ModelNew(nn.Module):
    """
    Simple model that performs a LeakyReLU activation.
    """

    def __init__(self, negative_slope: float=0.01):
        """
        Initializes the LeakyReLU module.

        Args:
            negative_slope (float, optional): The negative slope of the activation function. Defaults to 0.01.
        """
        super(ModelNew, self).__init__()
        self.negative_slope = negative_slope

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
    if 'negative_slope' in flat_state:
        state_kwargs['negative_slope'] = flat_state['negative_slope']
    else:
        state_kwargs['negative_slope'] = getattr(model, 'negative_slope')
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
# CUDA kernel – grid‑stride vectorized Leaky ReLU
# Improvements:
# 1. Grid-stride loop: drastically reduces kernel launch overhead.
# 2. float4 vectorization: ensures peak VRAM bandwidth utilization (128-bit loads/stores).
# 3. Memory Coalescing: alignment ensured by input.contiguous().
# 4. __restrict__ qualifiers: tells compiler there is no aliasing, improving load performance.
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector_types.h>

__global__ void leaky_relu_vectorized_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    float negative_slope,
    size_t n)
{
    // Grid stride: each thread handles multiple chunks of 4 elements.
    size_t idx = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 4;
    size_t stride = (size_t)blockDim.x * gridDim.x * 4;

    for (; idx + 4 <= n; idx += stride) {
        // Coalesced 128-bit load
        float4 vals = *reinterpret_cast<const float4*>(&input[idx]);

        // Apply Leaky ReLU
        vals.x = (vals.x > 0.0f) ? vals.x : vals.x * negative_slope;
        vals.y = (vals.y > 0.0f) ? vals.y : vals.y * negative_slope;
        vals.z = (vals.z > 0.0f) ? vals.z : vals.z * negative_slope;
        vals.w = (vals.w > 0.0f) ? vals.w : vals.w * negative_slope;

        // Coalesced 128-bit store
        *reinterpret_cast<float4*>(&output[idx]) = vals;
    }

    // Handle tail elements (if any)
    if (idx < n) {
        for (; idx < n; ++idx) {
            float val = input[idx];
            output[idx] = (val > 0.0f) ? val : val * negative_slope;
        }
    }
}

void leaky_relu_forward(torch::Tensor input, torch::Tensor output, float negative_slope) {
    const size_t n = input.numel();
    const int threads = 256;
    // Aim for enough blocks to saturate an RTX 2080Ti (e.g. 68 SMs * 4 blocks/SM = 272+)
    // Using 1024 blocks to safely cover grid-stride loop efficiency.
    const int blocks = 1024;

    leaky_relu_vectorized_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        negative_slope,
        n
    );
}
"""

# C++ Infrastructure
cpp_source = r"""
#include <torch/extension.h>

void leaky_relu_forward(torch::Tensor input, torch::Tensor output, float negative_slope);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("leaky_relu", &leaky_relu_forward, "Vectorized Leaky ReLU forward kernel");
}
"""

# Compile the extension
# Using --use_fast_math for faster hardware-level float operations
leaky_relu_ext = load_inline(
    name='leaky_relu_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, negative_slope):
    """
    Optimized functional_model using grid-stride vectorized CUDA kernel.
    """
    # 1. Ensure float32 (kernel requirement)
    if x.dtype != torch.float32:
        x = x.to(torch.float32)
    # 2. Ensure contiguous to allow float4 casts
    if not x.is_contiguous():
        x = x.contiguous()
    
    # Pre-allocate output
    output = torch.empty_like(x)
    
    # Launch kernel
    leaky_relu_ext.leaky_relu(x, output, float(negative_slope))
    
    return output

# --- Setup Constants ---
batch_size = 4096
dim = 393216

def get_init_inputs():
    return []

def get_inputs():
    x = torch.rand(batch_size, dim, device='cuda', dtype=torch.float32)
    return [x]
