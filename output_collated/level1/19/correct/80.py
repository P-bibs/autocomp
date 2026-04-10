# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_190701/code_30.py
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
# CUDA kernel: Each thread processes 8 float4 vectors (32 elements)
# This design balances memory coalescing, register usage, and overhead.
# -------------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_op_forward_kernel(const float* __restrict__ input,
                                        float* __restrict__ output,
                                        size_t n) {
    // Each thread processes 8 float4s = 32 elements.
    // This granularity provides enough work to hide latency and 
    // reduces the total grid size significantly compared to 1 thread/4 elements.
    const int elements_per_thread = 32;
    const int floats_per_vec = 4;
    
    size_t global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t start = global_thread_idx * elements_per_thread;

    // Use a fixed loop size to assist compiler unrolling
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        size_t idx = start + i * floats_per_vec;
        if (idx + 3 < n) {
            // Load 4 contiguous floats (1 float4)
            float4 in_vec = __ldg(reinterpret_cast<const float4*>(input) + (idx / 4));
            
            // Perform ReLU
            float4 out_vec;
            out_vec.x = fmaxf(in_vec.x, 0.0f);
            out_vec.y = fmaxf(in_vec.y, 0.0f);
            out_vec.z = fmaxf(in_vec.z, 0.0f);
            out_vec.w = fmaxf(in_vec.w, 0.0f);
            
            // Store results
            reinterpret_cast<float4*>(output)[idx / 4] = out_vec;
        } else if (idx < n) {
            // Boundary handling for vectors that don't fit perfectly
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                if (idx + j < n) {
                    float val = __ldg(input + idx + j);
                    output[idx + j] = fmaxf(val, 0.0f);
                }
            }
        }
    }
}

void fused_op_forward(torch::Tensor input, torch::Tensor output) {
    size_t n = input.numel();
    const int threads = 256;
    // Each block handles threads * 32 elements
    const int blocks = (n + (threads * 32) - 1) / (threads * 32);
    
    fused_op_forward_kernel<<<blocks, threads>>>(input.data_ptr<float>(),
                                                 output.data_ptr<float>(),
                                                 n);
}
"""

# -------------------------------------------------------------------------
# C++ binding
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(torch::Tensor input, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Optimized fused ReLU implementation");
}
"""

# -------------------------------------------------------------------------
# Build
# -------------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x: torch.Tensor) -> torch.Tensor:
    """
    In-place-like ReLU using vectorized memory access and increased 
    parallel work efficiency per thread.
    """
    output = torch.empty_like(x)
    fused_ext.fused_op(x, output)
    return output
