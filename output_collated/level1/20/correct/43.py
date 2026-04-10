# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_204858/code_16.py
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

# -------------------------------------------------------------------------
# Optimized CUDA kernel: Vectorized (float4) and coalesced
# -------------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void leaky_relu_kernel(const float* __restrict__ input,
                                  float* __restrict__ output,
                                  float negative_slope,
                                  int numel) {
    // Each thread processes a float4 (4 floats) to maximize throughput
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    
    // Process chunks of 4 sequentially to ensure coalescing
    for (int i = idx; i < numel; i += blockDim.x * gridDim.x * 4) {
        if (i + 3 < numel) {
            float4 in_vec = reinterpret_cast<const float4*>(input + i)[0];
            float4 out_vec;

            // Apply Leaky ReLU logic
            out_vec.x = in_vec.x > 0.0f ? in_vec.x : in_vec.x * negative_slope;
            out_vec.y = in_vec.y > 0.0f ? in_vec.y : in_vec.y * negative_slope;
            out_vec.z = in_vec.z > 0.0f ? in_vec.z : in_vec.z * negative_slope;
            out_vec.w = in_vec.w > 0.0f ? in_vec.w : in_vec.w * negative_slope;

            reinterpret_cast<float4*>(output + i)[0] = out_vec;
        } else {
            // Cleanup for remainder elements
            for (int k = i; k < i + 4 && k < numel; ++k) {
                float val = input[k];
                output[k] = val > 0.0f ? val : val * negative_slope;
            }
        }
    }
}

void launch_leaky_relu(torch::Tensor input, float negative_slope) {
    int numel = input.numel();
    int threads = 256;
    // Calculate grid size based on vectorized elements
    int blocks = (numel / 4 + threads - 1) / threads;
    // Cap blocks to prevent excessive overhead; 4096 is typically sufficient for full occupancy
    blocks = (blocks > 4096) ? 4096 : blocks;

    leaky_relu_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), 
        input.data_ptr<float>(), 
        negative_slope, 
        numel
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void launch_leaky_relu(torch::Tensor input, float negative_slope);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("leaky_relu", &launch_leaky_relu, "Leaky ReLU In-place");
}
"""

# Compile the extension
leaky_relu_ext = load_inline(
    name='leaky_relu_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, negative_slope):
    """
    Optimized in-place leaky ReLU.
    Ensures input is contiguous before calling the vectorized CUDA kernel.
    """
    if not x.is_contiguous():
        x = x.contiguous()
    
    leaky_relu_ext.leaky_relu(x, float(negative_slope))
    return x
