# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_225030/code_29.py
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
# Optimised CUDA kernel – grid‑stride loop + vectorized float4
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <algorithm>

__global__ void leaky_relu_grid_stride_kernel(
        const float* __restrict__ input,
        float* __restrict__ output,
        float negative_slope,
        const size_t total_vectors,
        const size_t remainder)
{
    // Grid-stride parameters
    const size_t stride = blockDim.x * gridDim.x;
    size_t vec_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Process all complete float4 vectors
    for (size_t i = vec_idx; i < total_vectors; i += stride) {
        // Load 4 consecutive floats through the texture cache
        const float4 in_vec = __ldg(reinterpret_cast<const float4*>(input) + i);

        // Branch-less Leaky ReLU: out = fmax(x,0) + slope * fmin(x,0)
        float4 out_vec;
        out_vec.x = fmaf(negative_slope, fminf(in_vec.x, 0.0f), fmaxf(in_vec.x, 0.0f));
        out_vec.y = fmaf(negative_slope, fminf(in_vec.y, 0.0f), fmaxf(in_vec.y, 0.0f));
        out_vec.z = fmaf(negative_slope, fminf(in_vec.z, 0.0f), fmaxf(in_vec.z, 0.0f));
        out_vec.w = fmaf(negative_slope, fminf(in_vec.w, 0.0f), fmaxf(in_vec.w, 0.0f));

        // Store the result
        reinterpret_cast<float4*>(output)[i] = out_vec;
    }

    // Handle the final <4 elements (remainder)
    if (remainder != 0 && vec_idx == 0) {
        const size_t base = total_vectors * 4;
        for (size_t j = 0; j < remainder; ++j) {
            const float val = input[base + j];
            output[base + j] = fmaf(negative_slope, fminf(val, 0.0f), fmaxf(val, 0.0f));
        }
    }
}

void leaky_relu_forward(torch::Tensor input, torch::Tensor output, float negative_slope) {
    const size_t n = input.numel();
    const size_t total_vectors = n / 4;
    const size_t remainder = n % 4;

    const int threads = 512;
    // Cap at 8192 blocks to avoid launch overhead while keeping GPU saturated
    const int max_blocks = 8192;
    const int blocks = (int)std::min((size_t)max_blocks, (total_vectors + threads - 1) / threads);

    leaky_relu_grid_stride_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        negative_slope,
        total_vectors,
        remainder
    );
}
"""

# ----------------------------------------------------------------------
# C++ binding
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>
void leaky_relu_forward(torch::Tensor input, torch::Tensor output, float negative_slope);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("leaky_relu", &leaky_relu_forward, "Vectorized Leaky ReLU (grid-stride)");
}
"""

# Build the extension
leaky_relu_ext = load_inline(
    name='leaky_relu_opt',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# Output buffer cache
_cached_output = None
_cached_shape = None

def functional_model(x, *, negative_slope):
    """
    Optimised Leaky ReLU using grid-stride loops and vectorized float4.
    """
    global _cached_output, _cached_shape

    if not x.is_contiguous():
        x = x.contiguous()
    if x.dtype != torch.float32:
        x = x.to(torch.float32)

    if _cached_output is None or x.shape != _cached_shape:
        _cached_output = torch.empty_like(x)
        _cached_shape = x.shape

    leaky_relu_ext.leaky_relu(x, _cached_output, float(negative_slope))
    return _cached_output
