# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_225030/code_20.py
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
# Optimized CUDA kernel with Vectorized Grid-Stride Loops
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector_types.h>

__global__ void leaky_relu_vectorized_grid_stride_kernel(
        const float* __restrict__ input,
        float* __restrict__ output,
        float negative_slope,
        size_t n) 
{
    // Process 4 elements at a time using float4 for max bandwidth via 128-bit loads
    const size_t n_vec = n / 4;
    size_t vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = vec_idx; i < n_vec; i += stride) {
        float4 in_vec = reinterpret_cast<const float4*>(input)[i];
        float4 out_vec;

        // Apply Leaky ReLU to 4 elements
        out_vec.x = (in_vec.x > 0.0f) ? in_vec.x : (in_vec.x * negative_slope);
        out_vec.y = (in_vec.y > 0.0f) ? in_vec.y : (in_vec.y * negative_slope);
        out_vec.z = (in_vec.z > 0.0f) ? in_vec.z : (in_vec.z * negative_slope);
        out_vec.w = (in_vec.w > 0.0f) ? in_vec.w : (in_vec.w * negative_slope);

        reinterpret_cast<float4*>(output)[i] = out_vec;
    }

    // Handle tail elements
    for (size_t i = n_vec * 4 + vec_idx; i < n; i += stride) {
        float val = input[i];
        output[i] = (val > 0.0f) ? val : (val * negative_slope);
    }
}

void leaky_relu_forward(torch::Tensor input, torch::Tensor output, float negative_slope) {
    const size_t n = input.numel();
    const int threads = 256;
    // Utilize 1024 blocks to saturate the 68 SMs of the 2080Ti
    const int blocks = 1024;

    leaky_relu_vectorized_grid_stride_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        negative_slope,
        n
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void leaky_relu_forward(torch::Tensor input, torch::Tensor output, float negative_slope);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("leaky_relu", &leaky_relu_forward, "Vectorized Grid-stride Leaky ReLU");
}
"""

# Compile the extension
leaky_relu_ext = load_inline(
    name='leaky_relu_optimized',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

_output_buffer = None

def functional_model(x, *, negative_slope):
    """
    Optimized functional model using vectorized grid-stride CUDA kernel.
    """
    global _output_buffer
    
    # Ensure inputs are in a format our kernel expects (contiguous float32)
    x_contig = x.contiguous()
    if x_contig.dtype != torch.float32:
        x_contig = x_contig.to(torch.float32)

    # Allocate output buffer once
    if _output_buffer is None or _output_buffer.shape != x_contig.shape:
        _output_buffer = torch.empty_like(x_contig)

    leaky_relu_ext.leaky_relu(x_contig, _output_buffer, float(negative_slope))
    
    return _output_buffer
