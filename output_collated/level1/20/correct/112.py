# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_220342/code_26.py
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

# High-performance CUDA Kernel using float4 vectorization and ILP
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void vectorized_leaky_relu_kernel(const float* __restrict__ input, 
                                             float* __restrict__ output, 
                                             float negative_slope, 
                                             size_t n) {
    // Each thread processes 4 elements via float4 for coalesced 128-bit loads
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    
    // Process as much as possible with 128-bit transactions
    if (idx + 3 < n) {
        float4 v = reinterpret_cast<const float4*>(&input[idx])[0];
        
        v.x = (v.x > 0.0f) ? v.x : v.x * negative_slope;
        v.y = (v.y > 0.0f) ? v.y : v.y * negative_slope;
        v.z = (v.z > 0.0f) ? v.z : v.z * negative_slope;
        v.w = (v.w > 0.0f) ? v.w : v.w * negative_slope;
        
        reinterpret_cast<float4*>(&output[idx])[0] = v;
    } else {
        // Handle tail with scalar logic
        for (int i = idx; i < n; ++i) {
            float val = input[i];
            output[i] = (val > 0.0f) ? val : val * negative_slope;
        }
    }
}

void leaky_relu_forward(torch::Tensor input, torch::Tensor output, float negative_slope) {
    const size_t n = input.numel();
    const int threads = 256;
    const int blocks = (n + 1023) / 1024; // 1024 elements per block (256 threads * 4)
    
    vectorized_leaky_relu_kernel<<<blocks, threads>>>(
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
    m.def("leaky_relu", &leaky_relu_forward, "Vectorized Leaky ReLU");
}
"""

leaky_relu_ext = load_inline(
    name='leaky_relu_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, negative_slope):
    """
    Optimized Leaky ReLU using vectorization (float4).
    This bypasses raw PyTorch loops, hitting peak memory bandwidth.
    """
    if x.dtype != torch.float32:
        x = x.to(torch.float32)
    # Ensure contiguous for coalesced memory access
    if not x.is_contiguous():
        x = x.contiguous()
    
    output = torch.empty_like(x)
    leaky_relu_ext.leaky_relu(x, output, float(negative_slope))
    return output
