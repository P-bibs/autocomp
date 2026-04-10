# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_190701/code_22.py
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
# Optimized CUDA kernel using Grid-Stride blocks and register-level vectorization
# -------------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <algorithm>

__global__ void fused_op_forward_kernel(const float* __restrict__ input,
                                        float* __restrict__ output,
                                        size_t n) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;

    // Process 4 elements per iteration per thread to utilize float4
    // We use a grid-stride loop to keep the GPU busy regardless of input size
    #pragma unroll
    for (size_t i = tid * 4; i < n; i += stride * 4) {
        if (i + 3 < n) {
            // Unaligned access is handled by hardware, but float4 loads are preferred
            float4 in_vec = *reinterpret_cast<const float4*>(input + i);
            
            float4 out_vec;
            out_vec.x = fmaxf(in_vec.x, 0.0f);
            out_vec.y = fmaxf(in_vec.y, 0.0f);
            out_vec.z = fmaxf(in_vec.z, 0.0f);
            out_vec.w = fmaxf(in_vec.w, 0.0f);
            
            *reinterpret_cast<float4*>(output + i) = out_vec;
        } else {
            // Handle remaining elements (the tail)
            #pragma unroll
            for (size_t j = 0; j < 4; ++j) {
                if (i + j < n) {
                    output[i + j] = fmaxf(input[i + j], 0.0f);
                }
            }
        }
    }
}

void fused_op_forward(torch::Tensor input, torch::Tensor output) {
    size_t n = input.numel();
    // RTX 2080Ti has 68 SMs. 128 threads/block gives good occupancy.
    const int threads = 128;
    const int blocks = 512; 
    
    fused_op_forward_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
}
"""

# -------------------------------------------------------------------------
# C++ binding (PYBIND11)
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(torch::Tensor input, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "High-performance fused ReLU kernel");
}
"""

# -------------------------------------------------------------------------
# Compile the extension
# -------------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math', '-arch=sm_75'],
    with_cuda=True
)

def functional_model(x: torch.Tensor) -> torch.Tensor:
    """
    Optimized in-place ReLU using a custom CUDA kernel with Grid-Stride loop
    and float4 vectorization.
    """
    if not x.is_cuda:
        return torch.nn.functional.relu(x)
        
    output = torch.empty_like(x)
    fused_ext.fused_op(x, output)
    return output
