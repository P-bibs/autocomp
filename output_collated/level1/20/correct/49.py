# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_204858/code_23.py
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

# Optimized CUDA Kernel for Vectorized Leaky ReLU
# Improvements:
# 1. Increased thread-level granularity to 16 elements per thread.
# 2. Used __restrict__ pointers to enable compiler optimizations (aliasing assumptions).
# 3. Used #pragma unroll to minimize branch overhead and maximize throughput.
# 4. Correctly aligned memory access to the widest possible vector (float4).
# 5. Adjusted block/grid strategy to ensure saturation of 2080Ti SMs.
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void leaky_relu_optimized_kernel(const float* __restrict__ input, float* __restrict__ output, 
                                             float negative_slope, size_t n) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    // Process 16 floats per thread (4 float4 vectors)
    for (size_t i = tid * 16; i < n; i += stride * 16) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            size_t idx = i + j * 4;
            if (idx + 3 < n) {
                // Vectorized Load
                float4 vals = *reinterpret_cast<const float4*>(&input[idx]);
                
                // Vectorized Calculation using ternary
                vals.x = (vals.x > 0.0f) ? vals.x : vals.x * negative_slope;
                vals.y = (vals.y > 0.0f) ? vals.y : vals.y * negative_slope;
                vals.z = (vals.z > 0.0f) ? vals.z : vals.z * negative_slope;
                vals.w = (vals.w > 0.0f) ? vals.w : vals.w * negative_slope;
                
                // Vectorized Store
                *reinterpret_cast<float4*>(&output[idx]) = vals;
            } else {
                // Handle edge cases where n is not a multiple of 4
                for (size_t k = idx; k < idx + 4 && k < n; ++k) {
                    output[k] = (input[k] > 0.0f) ? input[k] : input[k] * negative_slope;
                }
            }
        }
    }
}

void leaky_relu_forward(torch::Tensor input, torch::Tensor output, float negative_slope) {
    const size_t n = input.numel();
    const int threads = 256;
    // Saturate RTX 2080Ti (68 SMs). Using 1024 blocks provides enough work.
    const int blocks = std::min(1024, (int)((n + (16 * threads) - 1) / (16 * threads)));
    
    leaky_relu_optimized_kernel<<<blocks, threads>>>(
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
    m.def("leaky_relu", &leaky_relu_forward, "Vectorized Leaky ReLU forward kernel");
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

def functional_model(x, *, negative_slope):
    """
    Optimized functional_model using vectorized CUDA kernel.
    Handles memory efficiency via float32 contiguous alignment and 
    high-throughput per-thread processing.
    """
    if x.dtype != torch.float32:
        x = x.to(torch.float32)
    if not x.is_contiguous():
        x = x.contiguous()
    
    output = torch.empty_like(x)
    leaky_relu_ext.leaky_relu(x, output, float(negative_slope))
    return output
