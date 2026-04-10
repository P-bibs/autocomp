# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_202725/code_29.py
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

# ------------------------------------------------------------
#  Optimized Grid-stride vectorized CUDA kernel for Leaky ReLU
#  This implementation saturates memory bandwidth by minimizing
#  kernel launch overhead and ensuring optimal coalesced access.
# ------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector_types.h>

__global__ void leaky_relu_kernel_grid_stride(const float* __restrict__ input,
                                               float* __restrict__ output,
                                               float negative_slope,
                                               long num_elements) {
    // Total number of elements processed by all threads in one parallel iteration.
    // Each thread processes a float4 (4 floats).
    const long stride_elements = (long)blockDim.x * gridDim.x * 4;
    long idx = ((long)blockIdx.x * blockDim.x + threadIdx.x) * 4;

    // Process chunks of 4 floats as long as we are within bounds
    while (idx + 3 < num_elements) {
        // Use __ldg to force read-only cache usage for global memory loads
        float4 in_vec = __ldg((const float4*)(input + idx));

        float4 out_vec;
        out_vec.x = (in_vec.x > 0.0f) ? in_vec.x : in_vec.x * negative_slope;
        out_vec.y = (in_vec.y > 0.0f) ? in_vec.y : in_vec.y * negative_slope;
        out_vec.z = (in_vec.z > 0.0f) ? in_vec.z : in_vec.z * negative_slope;
        out_vec.w = (in_vec.w > 0.0f) ? in_vec.w : in_vec.w * negative_slope;

        reinterpret_cast<float4*>(output + idx)[0] = out_vec;

        // Grid-stride jump
        idx += stride_elements;
    }

    // Cleanup loop: Handle remaining elements if num_elements is not a multiple of 4 
    // or the grid-stride logic leaves elements.
    for (long i = idx; i < num_elements; ++i) {
        float val = __ldg(input + i);
        output[i] = (val > 0.0f) ? val : val * negative_slope;
    }
}

void leaky_relu_forward(torch::Tensor input, float negative_slope, torch::Tensor output) {
    const long num_elements = input.numel();
    const int threads_per_block = 256;
    // 2048 * 256 = 524,288 threads -> 2M element stride per iteration.
    // This provides high occupancy and optimal scheduling for RTX 2080Ti.
    const int blocks_per_grid = 2048;

    leaky_relu_kernel_grid_stride<<<blocks_per_grid, threads_per_block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        negative_slope,
        num_elements
    );
}
"""

# ------------------------------------------------------------
#  C++ binding via pybind11
# ------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void leaky_relu_forward(torch::Tensor input, float negative_slope, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("leaky_relu_cuda", &leaky_relu_forward, "Grid-stride vectorized Leaky ReLU");
}
"""

# Compile the extension with architecture-specific flags for Turing
fused_ext = load_inline(
    name='leaky_relu_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math', '-arch=sm_75'],
    with_cuda=True
)

def functional_model(x, *, negative_slope):
    """
    Optimized entry point. The kernel processes input in strides to minimize
    GPU overhead while maximizing throughput via float4 vectorization.
    """
    output = torch.empty_like(x)
    fused_ext.leaky_relu_cuda(x, float(negative_slope), output)
    return output

# Inputs setup
batch_size = 4096
dim = 393216

def get_init_inputs():
    return []

def get_inputs():
    x = torch.rand(batch_size, dim, device='cuda')
    return [x]
