# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_225030/code_27.py
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
# CUDA kernel – Optimized for 2080Ti with float2 vectorization
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void leaky_relu_vectorized_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    float negative_slope,
    size_t n)
{
    // Each thread processes 2 floats using float2 for 64-bit alignment/coalescing
    // 512 threads per block provides good occupancy balancing register pressure.
    size_t idx = (size_t)(blockIdx.x * blockDim.x + threadIdx.x) * 2;

    if (idx + 1 < n) {
        // Optimized vectorized load (64-bit memory access)
        float2 in_vec = reinterpret_cast<const float2*>(input)[idx / 2];
        float2 out_vec;

        // Efficient ternary operations (often compiled to predicated instructions)
        out_vec.x = (in_vec.x > 0.0f) ? in_vec.x : in_vec.x * negative_slope;
        out_vec.y = (in_vec.y > 0.0f) ? in_vec.y : in_vec.y * negative_slope;

        // Optimized vectorized store (64-bit memory access)
        reinterpret_cast<float2*>(output)[idx / 2] = out_vec;
    } else if (idx < n) {
        // Handle single tail element
        float val = input[idx];
        output[idx] = (val > 0.0f) ? val : val * negative_slope;
    }
}

void leaky_relu_forward(torch::Tensor input, torch::Tensor output, float negative_slope) {
    const size_t n = input.numel();
    const int block_size = 512;
    // Each thread covers 2 elements
    const int grid_size = (n + (2 * block_size) - 1) / (2 * block_size);
    
    leaky_relu_vectorized_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        negative_slope,
        n
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
    m.def("leaky_relu", &leaky_relu_forward, "Vectorized Leaky ReLU forward");
}
"""

# Compile the extension just-in-time
leaky_relu_ext = load_inline(
    name='leaky_relu_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, negative_slope):
    """
    Optimized functional_model:
    1. Ensures input is contiguous for coalesced access.
    2. Uses custom CUDA kernel with float2 vectorization.
    3. Minimizes thread divergence and register pressure.
    """
    if not x.is_contiguous():
        x = x.contiguous()
    
    if x.dtype != torch.float32:
        x = x.to(torch.float32)
    
    output = torch.empty_like(x)
    leaky_relu_ext.leaky_relu(x, output, float(negative_slope))
    return output

# Evaluation helpers
batch_size = 4096
dim = 393216

def get_init_inputs():
    return []

def get_inputs():
    # Returns inputs scaled for benchmarking
    x = torch.rand(batch_size, dim, device='cuda', dtype=torch.float32)
    return [x]
