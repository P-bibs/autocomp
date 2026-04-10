# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_185153/code_20.py
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
# Optimized CUDA Kernel
# Implements Grid-Stride Loop pattern and vectorized float4 memory access.
# This pattern provides persistent thread execution, reducing launch overhead 
# and allowing the compiler to perform optimal register allocation and ILP.
# -------------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void relu_optimized_kernel(const float* __restrict__ input, 
                                     float* __restrict__ output, 
                                     size_t n) {
    // 1. Grid-stride setup: define pointer access to float4
    const float4* in_vec = reinterpret_cast<const float4*>(input);
    float4* out_vec = reinterpret_cast<float4*>(output);
    
    size_t num_vectors = n / 4;
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    // 2. Main loop: Process float4 chunks
    // The grid-stride loop keeps the GPU saturated regardless of tensor size
    for (size_t i = tid; i < num_vectors; i += stride) {
        float4 val = in_vec[i];
        
        // ReLU on float4 components
        val.x = fmaxf(val.x, 0.0f);
        val.y = fmaxf(val.y, 0.0f);
        val.z = fmaxf(val.z, 0.0f);
        val.w = fmaxf(val.w, 0.0f);
        
        out_vec[i] = val;
    }
    
    // 3. Remainder handling: Process leftover elements (up to 3) 
    // Only performed by thread 0 to avoid synchronization issues
    if (tid == 0) {
        for (size_t i = num_vectors * 4; i < n; ++i) {
            output[i] = fmaxf(input[i], 0.0f);
        }
    }
}

void fused_op_forward(torch::Tensor input, torch::Tensor output) {
    const size_t n = input.numel();
    if (n == 0) return;
    
    // Heuristic: Use a sufficient amount of blocks to saturate the 2080Ti
    // 1024 threads per block is generally optimal for throughput-oriented kernels
    const int threads = 256;
    const int blocks = 480; // Enough to cover streaming multiprocessors on 2080Ti
    
    relu_optimized_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), n);
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_op_forward(torch::Tensor input, torch::Tensor output);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Grid-stride vectorized ReLU");
}
"""

# Build the inline extension
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math', '--ptxas-options=-v'],
    with_cuda=True
)

def functional_model(x: torch.Tensor) -> torch.Tensor:
    """
    Applies ReLU via optimized custom CUDA kernel.
    Ensures that contiguous memory is accessed for maximum bandwidth.
    """
    # Create empty output tensor
    output = torch.empty_like(x)
    # Ensure memory is contiguous for float4 casting
    if not x.is_contiguous():
        x = x.contiguous()
    
    fused_ext.fused_op(x, output)
    return output
