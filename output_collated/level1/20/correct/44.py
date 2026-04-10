# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_204858/code_17.py
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

# Optimized CUDA kernel:
# 1. Uses float4 for 128-bit memory throughput (coalesced 16-byte reads).
# 2. Uses __fselect (or fmaxf) to eliminate branch divergence in the hardware pipeline.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float leaky_relu_fsel(float x, float slope) {
    // fselect pattern: (x > 0 ? x : x * slope)
    // This allows the compiler to convert the ternary into a predicate instruction
    return (x > 0.0f) ? x : (x * slope);
}

__global__ void leaky_relu_vec4_kernel(const float* __restrict__ input, float* __restrict__ output, float slope, int numel) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    
    // Process 4 floats per thread using float4 vectorization
    if (idx + 3 < numel) {
        float4 in_vec = reinterpret_cast<const float4*>(input)[idx / 4];
        float4 out_vec;
        
        out_vec.x = leaky_relu_fsel(in_vec.x, slope);
        out_vec.y = leaky_relu_fsel(in_vec.y, slope);
        out_vec.z = leaky_relu_fsel(in_vec.z, slope);
        out_vec.w = leaky_relu_fsel(in_vec.w, slope);
        
        reinterpret_cast<float4*>(output)[idx / 4] = out_vec;
    } else {
        // Handle remainders (unlikely for large tensors but required for safety)
        for (int i = idx; i < numel; ++i) {
            float val = input[i];
            output[i] = (val > 0.0f) ? val : (val * slope);
        }
    }
}

void launch_leaky_relu(int numel, float* input, float* output, float slope) {
    const int block_size = 256;
    // Each thread handles 4 elements
    int grid_size = (numel / 4 + block_size - 1) / block_size;
    leaky_relu_vec4_kernel<<<grid_size, block_size>>>(input, output, slope, numel);
}
"""

cpp_source = r"""
#include <torch/extension.h>

void launch_leaky_relu(int numel, float* input, float* output, float negative_slope);

void leaky_relu_forward(torch::Tensor input, torch::Tensor output, float negative_slope) {
    launch_leaky_relu(input.numel(), input.data_ptr<float>(), output.data_ptr<float>(), negative_slope);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("leaky_relu_forward", &leaky_relu_forward, "Vectorized Leaky ReLU with predicate optimization");
}
"""

# Compile extension
leaky_relu_ext = load_inline(
    name='leaky_relu_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, negative_slope):
    """
    Optimized Leaky ReLU using float4 vectorization and branchless predicates.
    """
    if not x.is_contiguous():
        x = x.contiguous()
    output = torch.empty_like(x)
    leaky_relu_ext.leaky_relu_forward(x, output, float(negative_slope))
    return output

def get_init_inputs():
    return []

def get_inputs():
    batch_size, dim = 4096, 393216
    x = torch.rand(batch_size, dim, device='cuda', dtype=torch.float32)
    return [x]
