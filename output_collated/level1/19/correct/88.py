# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_192155/code_24.py
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

# -------------------------------------------------------------------------
# CUDA source – Vectorized ReLU kernel
# Using float4 enables 128-bit memory transactions per thread, 
# significantly improving memory bandwidth saturation.
# -------------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void relu_vec_kernel(const float* __restrict__ input,
                                 float* __restrict__ output,
                                 size_t n_vec) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_vec) {
        // Load 4 floats at once (128-bit wide load)
        float4 in_vec = reinterpret_cast<const float4*>(input)[idx];
        
        // Execute ReLU vectorized
        float4 out_vec;
        out_vec.x = fmaxf(in_vec.x, 0.0f);
        out_vec.y = fmaxf(in_vec.y, 0.0f);
        out_vec.z = fmaxf(in_vec.z, 0.0f);
        out_vec.w = fmaxf(in_vec.w, 0.0f);
        
        // Store 4 floats at once
        reinterpret_cast<float4*>(output)[idx] = out_vec;
    }
}

void fused_op_forward(torch::Tensor input, torch::Tensor output) {
    size_t n = input.numel();
    // Use float4, so we process n/4 blocks
    size_t n_vec = n / 4;
    
    // 256 threads per block is generally the sweet spot for Turing
    // balancing occupancy and register consumption.
    const int threads = 256;
    const int blocks = (n_vec + threads - 1) / threads;
    
    if (n_vec > 0) {
        relu_vec_kernel<<<blocks, threads>>>(
            input.data_ptr<float>(), output.data_ptr<float>(), n_vec);
    }
    
    // Handle tail if n % 4 != 0
    if (n % 4 != 0) {
        size_t start = (n_vec) * 4;
        for (size_t i = start; i < n; ++i) {
            output.data_ptr<float>()[i] = fmaxf(input.data_ptr<float>()[i], 0.0f);
        }
    }
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_op_forward(torch::Tensor input, torch::Tensor output);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Vectorized ReLU kernel");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x: torch.Tensor) -> torch.Tensor:
    """Applies ReLU using the vectorized fused kernel."""
    output = torch.empty_like(x)
    fused_ext.fused_op(x, output)
    return output

# Benchmark inputs
batch_size = 4096
dim = 393216

def get_init_inputs(): return []
def get_inputs(): return [torch.rand(batch_size, dim, device='cuda')]
