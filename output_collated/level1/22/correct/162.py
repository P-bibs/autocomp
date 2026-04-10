# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_012911/code_19.py
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

# Optimized CUDA kernel:
# 1. Grid-stride loop ensures balance across any N.
# 2. float4 vectorization maximizes memory throughput.
# 3. Uses __restrict__ to allow compiler to optimize pointer aliasing.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector_types.h>

__global__ void tanh_optimized_kernel(const float* __restrict__ x, float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    // We process float4 vectors to maximize memory bandwidth (128-bit loads)
    // First, process the main body of vectors
    size_t vec_n = (n / 4) * 4;
    
    for (size_t i = idx * 4; i < vec_n; i += stride * 4) {
        float4 vec_x = reinterpret_cast<const float4*>(x)[i / 4];
        float4 vec_out;
        vec_out.x = tanhf(vec_x.x);
        vec_out.y = tanhf(vec_x.y);
        vec_out.z = tanhf(vec_x.z);
        vec_out.w = tanhf(vec_x.w);
        reinterpret_cast<float4*>(out)[i / 4] = vec_out;
    }
    
    // Handle remaining elements (if any)
    for (size_t i = vec_n + idx; i < n; i += stride) {
        out[i] = tanhf(x[i]);
    }
}

void fused_tanh_forward(torch::Tensor x, torch::Tensor out) {
    const size_t n = x.numel();
    if (n == 0) return;

    // Use occupancy-friendly block size
    const int threads = 256;
    // Aim for enough blocks to fill the GPU (2080Ti has 68 SMs)
    // 2048 blocks is a safe heuristic for large reductions/maps
    const int max_blocks = 2048;
    const int blocks = (n / 4 + threads - 1) / threads;
    const int grid = std::min(max_blocks, std::max(1, blocks));
    
    tanh_optimized_kernel<<<grid, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), n);
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_tanh_forward(torch::Tensor x, torch::Tensor out);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_tanh", &fused_tanh_forward, "Optimized Tanh forward");
}
"""

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
    Ensures input is contiguous to maximize memory coalescing.
    """
    if not x.is_contiguous():
        x = x.contiguous()
    out = torch.empty_like(x)
    fused_ext.fused_tanh(x, out)
    return out
