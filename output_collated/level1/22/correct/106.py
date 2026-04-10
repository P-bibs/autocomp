# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_004356/code_19.py
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

# Vectorized CUDA kernel using optimized grid-stride loops
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector_types.h>

__global__ void tanh_vec4_kernel(const float* __restrict__ x, float* __restrict__ out, size_t n) {
    // Grid-stride loop: each thread processes multiple 4-element chunks
    // Use __ldg for read-only x to potentially leverage texture cache
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    for (size_t idx = tid * 4; idx < n; idx += stride * 4) {
        if (idx + 3 < n) {
            float4 vec_x = reinterpret_cast<const float4*>(x)[idx / 4];
            float4 vec_out;
            vec_out.x = tanhf(vec_x.x);
            vec_out.y = tanhf(vec_x.y);
            vec_out.z = tanhf(vec_x.z);
            vec_out.w = tanhf(vec_x.w);
            reinterpret_cast<float4*>(out)[idx / 4] = vec_out;
        } else {
            for (size_t i = idx; i < n && i < idx + 4; ++i) {
                out[i] = tanhf(x[i]);
            }
        }
    }
}

void fused_tanh_forward(torch::Tensor x, torch::Tensor out) {
    const size_t n = x.numel();
    const int threads = 256;
    // Calculate grid size based on device occupancy considerations
    // 4 elements processed per thread, so we divide total n by 4
    const int num_vectors = (n + 3) / 4;
    const int blocks = std::min((num_vectors + threads - 1) / threads, 2048);
    
    tanh_vec4_kernel<<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), n);
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_tanh_forward(torch::Tensor x, torch::Tensor out);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_tanh", &fused_tanh_forward, "Vectorized Tanh forward with optimized grid-stride loop");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_tanh',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x):
    """
    Computes tanh(x) using a custom vectorized CUDA kernel.
    Ensures that input is contiguous for optimal float4 memory access.
    """
    if not x.is_contiguous():
        x = x.contiguous()
    out = torch.empty_like(x)
    fused_ext.fused_tanh(x, out)
    return out

# Configuration
batch_size = 4096
dim = 393216

def get_init_inputs():
    return []

def get_inputs():
    x = torch.rand(batch_size, dim, device='cuda', dtype=torch.float32)
    return [x]
