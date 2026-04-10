# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_190701/code_15.py
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

# Optimization: Increase vectorization width from 4 to 8 elements per thread
# This reduces kernel launch overhead and improves occupancy
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_op_forward_kernel(const float* __restrict__ input, float* __restrict__ output, size_t n) {
    size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    
    // Process 8 elements using two float4 vectors for maximum memory throughput
    if (idx + 7 < n) {
        // Load two float4 vectors
        float4 in_vec1 = reinterpret_cast<const float4*>(input)[idx / 4];
        float4 in_vec2 = reinterpret_cast<const float4*>(input)[(idx + 4) / 4];
        
        // Apply ReLU activation (branchless)
        float4 out_vec1;
        out_vec1.x = fmaxf(in_vec1.x, 0.0f);
        out_vec1.y = fmaxf(in_vec1.y, 0.0f);
        out_vec1.z = fmaxf(in_vec1.z, 0.0f);
        out_vec1.w = fmaxf(in_vec1.w, 0.0f);
        
        float4 out_vec2;
        out_vec2.x = fmaxf(in_vec2.x, 0.0f);
        out_vec2.y = fmaxf(in_vec2.y, 0.0f);
        out_vec2.z = fmaxf(in_vec2.z, 0.0f);
        out_vec2.w = fmaxf(in_vec2.w, 0.0f);
        
        // Store results
        reinterpret_cast<float4*>(output)[idx / 4] = out_vec1;
        reinterpret_cast<float4*>(output)[(idx + 4) / 4] = out_vec2;
    } 
    // Handle tail: Process remaining elements (1-7) with a loop
    else {
        for (int i = 0; i < 8; ++i) {
            if (idx + i < n) {
                float val = input[idx + i];
                output[idx + i] = fmaxf(val, 0.0f);
            }
        }
    }
}

void fused_op_forward(torch::Tensor input, torch::Tensor output) {
    size_t n = input.numel();
    const int threads = 256;
    // Process 8 elements per thread, so divide by 8
    const int blocks = (n / 8 + threads - 1) / threads;
    fused_op_forward_kernel<<<blocks, threads>>>(input.data_ptr<float>(), output.data_ptr<float>(), n);
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_op_forward(torch::Tensor input, torch::Tensor output);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Optimized Vectorized ReLU (8 elements/thread)");
}
"""

fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math', '-arch=sm_75'],
    with_cuda=True
)

def functional_model(x):
    output = torch.empty_like(x)
    fused_ext.fused_op(x, output)
    return output

# --- Evaluation setup ---
batch_size = 4096
dim = 393216

def get_init_inputs():
    return []

def get_inputs():
    x = torch.rand(batch_size, dim, device='cuda')
    return [x]
