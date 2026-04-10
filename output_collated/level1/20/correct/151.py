# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_225030/code_23.py
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
# CUDA Kernel Optimization Strategy:
# 1. Thread coarse-graining: Each thread processes 8 floats (via float4) to 
#    increase instruction-level parallelism and hide memory latency.
# 2. Occupancy: Tuning block size to 128/256 for the RT2080Ti (Turing) 
#    to maximize throughput for memory-bound ops.
# 3. Stream integration: Using at::cuda::getCurrentCUDAStream() to ensure 
#    zero-copy integration within PyTorch's dependency graph.
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

__global__ void leaky_relu_vectorized_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    float negative_slope,
    size_t n)
{
    // Process 8 floats per thread (2x float4) to maximize throughput
    size_t idx = (size_t)(blockIdx.x * blockDim.x + threadIdx.x) * 8;

    if (idx + 7 < n) {
        float4 v1 = reinterpret_cast<const float4*>(input)[(idx) / 4];
        float4 v2 = reinterpret_cast<const float4*>(input)[(idx + 4) / 4];

        auto apply_leaky = [negative_slope](float v) {
            return (v > 0.0f) ? v : v * negative_slope;
        };

        v1.x = apply_leaky(v1.x); v1.y = apply_leaky(v1.y);
        v1.z = apply_leaky(v1.z); v1.w = apply_leaky(v1.w);
        v2.x = apply_leaky(v2.x); v2.y = apply_leaky(v2.y);
        v2.z = apply_leaky(v2.z); v2.w = apply_leaky(v2.w);

        reinterpret_cast<float4*>(output)[(idx) / 4] = v1;
        reinterpret_cast<float4*>(output)[(idx + 4) / 4] = v2;
    } else {
        // Scalar tail handling
        for (size_t i = idx; i < n; ++i) {
            float val = input[i];
            output[i] = (val > 0.0f) ? val : val * negative_slope;
        }
    }
}

void leaky_relu_forward(torch::Tensor input, torch::Tensor output, float negative_slope) {
    const size_t n = input.numel();
    const int block_size = 256;
    // Each thread handles 8 elements
    const int grid_size = (n / 8 + block_size - 1) / block_size;
    
    leaky_relu_vectorized_kernel<<<grid_size, block_size, 0, at::cuda::getCurrentCUDAStream()>>>(
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
    m.def("leaky_relu", &leaky_relu_forward, "Vectorized Leaky ReLU forward");
}
"""

# Compile extension
leaky_relu_ext = load_inline(
    name='leaky_relu_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math', '-arch=sm_75'],
    with_cuda=True
)

def functional_model(x, *, negative_slope):
    """
    Optimized functional_model for Leaky ReLU on RTX 2080Ti.
    """
    # Ensure memory is contiguous for coalesced float4/float8 loads
    if not x.is_contiguous():
        x = x.contiguous()
    
    # Ensure buffer is float32
    if x.dtype != torch.float32:
        x = x.to(torch.float32)
        
    output = torch.empty_like(x)
    leaky_relu_ext.leaky_relu(x, output, float(negative_slope))
    return output

# Evaluation helpers
def get_inputs():
    batch_size = 4096
    dim = 393216
    x = torch.rand(batch_size, dim, device='cuda', dtype=torch.float32)
    return [x]
