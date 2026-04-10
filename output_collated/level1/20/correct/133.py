# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_222822/code_30.py
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
# CUDA kernel – Optimized for higher occupancy on RTX 2080 Ti
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Kernel with vectorised float4 loads/stores and occupancy hints
// __launch_bounds__(256) ensures the compiler keeps register usage low
// enough to support at least 256 threads per SM effectively.
__global__ __launch_bounds__(256) void leaky_relu_vectorized_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    float negative_slope,
    size_t n) 
{
    // Each thread handles 4 contiguous elements (128-bit load/store)
    size_t idx = (static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 4;

    if (idx + 4 <= n) {
        // Load, compute, and store as float4 to maximize bandwidth
        float4 vals = *reinterpret_cast<const float4*>(&input[idx]);

        float4 result;
        result.x = (vals.x > 0.0f) ? vals.x : __fmul_rn(vals.x, negative_slope);
        result.y = (vals.y > 0.0f) ? vals.y : __fmul_rn(vals.y, negative_slope);
        result.z = (vals.z > 0.0f) ? vals.z : __fmul_rn(vals.z, negative_slope);
        result.w = (vals.w > 0.0f) ? vals.w : __fmul_rn(vals.w, negative_slope);

        *reinterpret_cast<float4*>(&output[idx]) = result;
    } else {
        // Handle remaining elements with a vectorized-friendly scalar loop
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            size_t pos = idx + i;
            if (pos < n) {
                float val = input[pos];
                output[pos] = (val > 0.0f) ? val : __fmul_rn(val, negative_slope);
            }
        }
    }
}

void leaky_relu_forward(torch::Tensor input, torch::Tensor output, float negative_slope) {
    const size_t n = input.numel();
    const int threads = 256;
    // Calculate grid size based on 4 elements per thread
    const int blocks = (static_cast<int>((n + 3) / 4) + threads - 1) / threads;

    leaky_relu_vectorized_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        negative_slope,
        n
    );
}
"""

# ----------------------------------------------------------------------
# C++ binding (PyBind11)
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void leaky_relu_forward(torch::Tensor input, torch::Tensor output, float negative_slope);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("leaky_relu", &leaky_relu_forward, "Vectorized Leaky ReLU");
}
"""

# ----------------------------------------------------------------------
# Extension Compilation
# ----------------------------------------------------------------------
leaky_relu_ext = load_inline(
    name='leaky_relu_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# ----------------------------------------------------------------------
# Functional Model
# ----------------------------------------------------------------------
def functional_model(x, *, negative_slope):
    """
    Optimized functional_model:
    Ensures input is float32 and contiguous, then invokes the custom
    CUDA kernel optimized for SM occupancy and registry usage.
    """
    if x.dtype != torch.float32:
        x = x.to(torch.float32)
    if not x.is_contiguous():
        x = x.contiguous()

    output = torch.empty_like(x)
    leaky_relu_ext.leaky_relu(x, output, float(negative_slope))
    return output

# --- Setup for evaluation harness ---
batch_size = 4096
dim = 393216

def get_init_inputs():
    return []

def get_inputs():
    x = torch.rand(batch_size, dim, device='cuda', dtype=torch.float32)
    return [x]
