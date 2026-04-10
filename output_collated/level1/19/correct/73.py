# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_190701/code_23.py
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

# Optimized CUDA kernel using vectorized loads (float4) combined with a grid-stride loop.
# This ensures that we saturate memory bandwidth while maintaining clean, branchless vector processing.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_op_forward_kernel(const float* __restrict__ input, float* __restrict__ output, size_t n) {
    size_t vec_n = n / 4;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    // Vectorized processing loop
    for (size_t i = idx; i < vec_n; i += stride) {
        float4 in_vec = reinterpret_cast<const float4*>(input)[i];
        float4 out_vec;
        out_vec.x = fmaxf(in_vec.x, 0.0f);
        out_vec.y = fmaxf(in_vec.y, 0.0f);
        out_vec.z = fmaxf(in_vec.z, 0.0f);
        out_vec.w = fmaxf(in_vec.w, 0.0f);
        reinterpret_cast<float4*>(output)[i] = out_vec;
    }

    // Scalar processing for the tail (elements n % 4)
    size_t remaining_start = vec_n * 4;
    for (size_t i = remaining_start + idx; i < n; i += stride) {
        output[i] = fmaxf(input[i], 0.0f);
    }
}

void fused_op_forward(torch::Tensor input, torch::Tensor output) {
    size_t n = input.numel();
    const int threads = 256;
    // Maximize occupancy: use a sufficient number of blocks to fill the GPU
    // 1024 blocks is a standard heuristic for high-throughput memory-bound ops
    const int blocks = 1024;
    
    fused_op_forward_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        n
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_op_forward(torch::Tensor input, torch::Tensor output);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Optimized Grid-Stride Coalesced ReLU");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math', '-arch=sm_75'],
    with_cuda=True
)

def functional_model(x):
    """
    Optimized functional interface for ReLU.
    x: torch.Tensor on CUDA (float32)
    """
    output = torch.empty_like(x)
    fused_ext.fused_op(x, output)
    return output
