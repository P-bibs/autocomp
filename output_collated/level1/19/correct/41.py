# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_185153/code_18.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = []
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = []
REQUIRED_FLAT_STATE_NAMES = []


class ModelNew(nn.Module):
    """
    Simple model that performs a ReLU activation.
    """

    def __init__(self):
        super(ModelNew, self).__init__()

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

# CUDA Kernel for high-performance Vectorized ReLU
# Using Grid-Stride Loops to balance occupancy and launch overhead.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void relu_optimized_kernel(const float* __restrict__ input, float* __restrict__ output, size_t num_elements) {
    // Each thread processes multiple float4 segments (Grid-Stride Loop)
    // This allows us to scale for any input size while maintaining coalescence.
    size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    size_t stride = blockDim.x * gridDim.x * 4;

    for (size_t i = idx; i < num_elements; i += stride) {
        if (i + 3 < num_elements) {
            float4 val = reinterpret_cast<const float4*>(input)[i / 4];
            
            val.x = fmaxf(0.0f, val.x);
            val.y = fmaxf(0.0f, val.y);
            val.z = fmaxf(0.0f, val.z);
            val.w = fmaxf(0.0f, val.w);
            
            reinterpret_cast<float4*>(output)[i / 4] = val;
        } else {
            // Remainder handling for non-aligned lengths
            for (size_t j = i; j < num_elements && j < i + 4; ++j) {
                output[j] = fmaxf(0.0f, input[j]);
            }
        }
    }
}

void relu_optimized_launch(torch::Tensor input, torch::Tensor output) {
    const size_t num_elements = input.numel();
    const int threads = 256;
    
    // Heuristic: Use enough blocks to saturate all SMs. 
    // 32768 is a safe upper bound to ensure high occupancy on modern GPUs.
    const int blocks = std::min((size_t)32768, (num_elements + 1023) / 1024);
    
    relu_optimized_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        num_elements
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void relu_optimized_launch(torch::Tensor input, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("relu_optimized", &relu_optimized_launch, "Optimized Grid-Stride Vectorized ReLU");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_relu_v2',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x):
    """
    Optimized ReLU implementation.
    """
    output = torch.empty_like(x)
    fused_ext.relu_optimized(x, output)
    return output

# Evaluation setup constants
batch_size = 4096
dim = 393216

def get_init_inputs():
    return []

def get_inputs():
    return [torch.rand(batch_size, dim, device='cuda')]
