# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_010905/code_9.py
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

# Optimized CUDA kernel using grid-stride loops
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector_types.h>

__global__ void tanh_optimized_kernel(const float* __restrict__ x, float* __restrict__ out, size_t n) {
    // Grid-stride loop: each thread processes multiple chunks across the entire array
    size_t stride = gridDim.x * blockDim.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Process 4 elements per iteration when possible (vectorized path)
    size_t vec_n = (n / 4) * 4;
    
    // Main vectorized path using grid-stride loop
    for (size_t i = idx * 4; i < vec_n; i += stride * 4) {
        float4 vec_x = reinterpret_cast<const float4*>(x)[i / 4];
        float4 vec_out;
        vec_out.x = tanhf(vec_x.x);
        vec_out.y = tanhf(vec_x.y);
        vec_out.z = tanhf(vec_x.z);
        vec_out.w = tanhf(vec_x.w);
        reinterpret_cast<float4*>(out)[i / 4] = vec_out;
    }
    
    // Handle remaining elements (0-3 elements) - all threads participate
    for (size_t i = vec_n + idx; i < n; i += stride) {
        out[i] = tanhf(x[i]);
    }
}

void fused_tanh_forward(torch::Tensor x, torch::Tensor out) {
    const size_t n = x.numel();
    const int threads = 256;
    // Launch sufficient blocks to keep GPU busy
    // For RTX 2080Ti, we can have up to 65535 blocks
    int blocks = (n / (threads * 4) + threads - 1) / threads;
    blocks = min(blocks, 65535);  // Cap at hardware limit
    if (blocks == 0) blocks = 1;  // Ensure at least 1 block
    
    tanh_optimized_kernel<<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), n);
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_tanh_forward(torch::Tensor x, torch::Tensor out);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_tanh", &fused_tanh_forward, "Optimized Grid-Stride Tanh forward");
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
    Optimized tanh operation using grid-stride loops.
    Ensures input is contiguous for safe float4 reinterpret_cast.
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
