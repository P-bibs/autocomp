# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_225030/code_4.py
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
# Constants
# ----------------------------------------------------------------------
batch_size = 4096
dim = 393216

def get_init_inputs():
    return []

def get_inputs():
    # Create a random input tensor of the required shape and dtype
    x = torch.rand(batch_size, dim, device='cuda', dtype=torch.float32)
    return [x]

# ----------------------------------------------------------------------
# Optimised CUDA kernel (Grid-stride loop + vectorized loads)
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void leaky_relu_grid_stride_kernel(
        const float* __restrict__ input,
        float* __restrict__ output,
        float negative_slope,
        size_t n) 
{
    // Grid-stride loop: each thread processes multiple elements
    const size_t stride = blockDim.x * gridDim.x;
    
    // Vectorized path: process 4 elements per iteration when possible
    size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    const size_t vec_n = n / 4 * 4;  // Round down to nearest multiple of 4
    
    // Vectorized loop
    for (; idx < vec_n; idx += stride * 4) {
        // Load 4 consecutive floats as a float4 via texture cache
        const float4 in_vec = __ldg(reinterpret_cast<const float4*>(input) + idx/4);
        
        // Apply branchless Leaky ReLU using fmaf for better performance
        float4 out_vec;
        out_vec.x = fmaf(negative_slope, fminf(in_vec.x, 0.0f), fmaxf(in_vec.x, 0.0f));
        out_vec.y = fmaf(negative_slope, fminf(in_vec.y, 0.0f), fmaxf(in_vec.y, 0.0f));
        out_vec.z = fmaf(negative_slope, fminf(in_vec.z, 0.0f), fmaxf(in_vec.z, 0.0f));
        out_vec.w = fmaf(negative_slope, fminf(in_vec.w, 0.0f), fmaxf(in_vec.w, 0.0f));
        
        // Store result
        reinterpret_cast<float4*>(output)[idx/4] = out_vec;
    }
    
    // Handle remaining elements scalarly
    idx /= 4; // Convert back to element index
    for (size_t i = idx * 4 + threadIdx.x; i < n; i += stride) {
        const float val = input[i];
        output[i] = fmaf(negative_slope, fminf(val, 0.0f), fmaxf(val, 0.0f));
    }
}

void leaky_relu_forward(torch::Tensor input, torch::Tensor output, float negative_slope) {
    const size_t n = input.numel();
    const int threads = 256;
    const int blocks = min(1024, (int)((n + threads - 1) / threads)); // Adaptive block count

    leaky_relu_grid_stride_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        negative_slope,
        n
    );
}
"""

# ----------------------------------------------------------------------
# C++ binding (PYBIND11)
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>
void leaky_relu_forward(torch::Tensor input, torch::Tensor output, float negative_slope);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("leaky_relu", &leaky_relu_forward, "Grid-stride vectorized Leaky ReLU with __ldg");
}
"""

# ----------------------------------------------------------------------
# Build the inline extension
# ----------------------------------------------------------------------
leaky_relu_ext = load_inline(
    name='leaky_relu_opt',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# ----------------------------------------------------------------------
# Global output buffer (re‑used across calls)
# ----------------------------------------------------------------------
_output_buffer = None

# ----------------------------------------------------------------------
# Functional model – entry point used by the evaluator
# ----------------------------------------------------------------------
def functional_model(x, *, negative_slope):
    """
    Optimised functional model using a custom CUDA kernel.
    The kernel uses:
      * Grid-stride loops for better scalability and hardware utilization,
      * Vectorized memory accesses (__ldg + float4) for higher bandwidth utilization,
      * Branch‑less Leaky ReLU (fmaxf / fminf + fmaf) for lower latency.
    """
    global _output_buffer

    # Ensure the input is contiguous and of the correct type
    if not x.is_contiguous():
        x = x.contiguous()
    if x.dtype != torch.float32:
        x = x.to(torch.float32)

    # Allocate the output buffer once (or re‑allocate if shape changes)
    if _output_buffer is None or _output_buffer.shape != x.shape:
        _output_buffer = torch.empty_like(x)

    # Launch the optimised kernel; result is stored in the reusable buffer
    leaky_relu_ext.leaky_relu(x, _output_buffer, float(negative_slope))

    # Return the buffer (the caller only reads its values)
    return _output_buffer
