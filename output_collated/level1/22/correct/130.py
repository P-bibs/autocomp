# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_010905/code_13.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = []
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = []
REQUIRED_FLAT_STATE_NAMES = []


class ModelNew(nn.Module):
    """
    Simple model that performs a Tanh activation.
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

# Optimized CUDA kernel with the following improvements:
# 1. Better grid/block sizing for RTX 2080Ti
# 2. More efficient remainder handling
# 3. Memory coalescing improvements
# 4. Better resource utilization
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector_types.h>

__global__ void tanh_optimized_kernel(const float* __restrict__ x, float* __restrict__ out, size_t n) {
    // Calculate global thread index
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    
    // Process elements with vectorized loads where possible
    size_t vec_n = (n / 4) * 4;
    
    // Vectorized processing loop
    for (size_t idx = tid * 4; idx < vec_n; idx += stride * 4) {
        float4 vec_x = reinterpret_cast<const float4*>(x)[idx / 4];
        float4 vec_out;
        vec_out.x = tanhf(vec_x.x);
        vec_out.y = tanhf(vec_x.y);
        vec_out.z = tanhf(vec_x.z);
        vec_out.w = tanhf(vec_x.w);
        reinterpret_cast<float4*>(out)[idx / 4] = vec_out;
    }
    
    // Handle remaining elements
    for (size_t i = tid + vec_n; i < n; i += stride) {
        out[i] = tanhf(x[i]);
    }
}

void fused_tanh_forward(torch::Tensor x, torch::Tensor out) {
    const size_t n = x.numel();
    
    // RTX 2080Ti has 46 SMs, optimal block size is usually 256 or 512
    const int threads_per_block = 256;
    
    // For good occupancy on RTX 2080Ti, we want enough blocks to fill SMs
    // Aim for at least 2-4 blocks per SM (46 SMs * 4 = 184 blocks)
    const int blocks_per_sm = 4;
    const int multi_processor_count = 46; // RTX 2080Ti
    const int max_blocks = multi_processor_count * blocks_per_sm;
    
    // Calculate required blocks for vectorized elements + remainder
    const size_t vec_n = (n / 4) * 4;
    const int blocks_for_vec = (vec_n / 4 + threads_per_block - 1) / threads_per_block;
    const int blocks_for_remainder = (n - vec_n + threads_per_block - 1) / threads_per_block;
    const int total_blocks_needed = blocks_for_vec + blocks_for_remainder;
    
    // Cap blocks at reasonable maximum for the hardware
    const int blocks = min(total_blocks_needed, max_blocks);
    
    tanh_optimized_kernel<<<blocks, threads_per_block>>>(x.data_ptr<float>(), out.data_ptr<float>(), n);
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_tanh_forward(torch::Tensor x, torch::Tensor out);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_tanh", &fused_tanh_forward, "Optimized Vectorized Tanh forward");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_tanh_opt',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x):
    """
    Optimized tanh operation.
    Ensures input is contiguous to allow safe reinterpret_cast to float4.
    """
    if not x.is_contiguous():
        x = x.contiguous()
    out = torch.empty_like(x)
    fused_ext.fused_tanh(x, out)
    return out

def get_inputs():
    # Standard input generation
    batch_size = 4096
    dim = 393216
    x = torch.rand(batch_size, dim, device='cuda', dtype=torch.float32)
    return [x]
