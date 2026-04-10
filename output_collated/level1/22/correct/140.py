# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_010905/code_27.py
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

# Optimized CUDA kernel: Two-pass strategy to ensure perfect coalescing and zero divergence
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector_types.h>

// Process main body using float4 (128-bit loads)
__global__ void tanh_vec4_kernel(const float* __restrict__ x, float* __restrict__ out, size_t n_vec4) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_vec4) {
        float4 vec_x = reinterpret_cast<const float4*>(x)[idx];
        float4 vec_out;
        vec_out.x = tanhf(vec_x.x);
        vec_out.y = tanhf(vec_x.y);
        vec_out.z = tanhf(vec_x.z);
        vec_out.w = tanhf(vec_x.w);
        reinterpret_cast<float4*>(out)[idx] = vec_out;
    }
}

// Process remainder (non-aligned trailing elements)
__global__ void tanh_scalar_kernel(const float* __restrict__ x, float* __restrict__ out, 
                                    size_t offset, size_t n_rem) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_rem) {
        out[offset + idx] = tanhf(x[offset + idx]);
    }
}

void fused_tanh_forward(torch::Tensor x, torch::Tensor out) {
    const size_t n = x.numel();
    const size_t n_vec4 = n / 4;
    const size_t n_rem = n % 4;
    
    // Using 128 threads for reasonable occupancy on Turing architecture
    const int threads = 128;
    
    if (n_vec4 > 0) {
        const int blocks = (int)((n_vec4 + threads - 1) / threads);
        tanh_vec4_kernel<<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), n_vec4);
    }
    
    if (n_rem > 0) {
        // Remainder is very small (1-3 elements), so 1 block is sufficient
        tanh_scalar_kernel<<<1, n_rem>>>(x.data_ptr<float>(), out.data_ptr<float>(), n_vec4 * 4, n_rem);
    }
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_tanh_forward(torch::Tensor x, torch::Tensor out);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_tanh", &fused_tanh_forward, "Optimized Tanh forward");
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
    # Ensure input is contiguous for float4 vectorization
    if not x.is_contiguous():
        x = x.contiguous()
    out = torch.empty_like(x)
    fused_ext.fused_tanh(x, out)
    return out

# Configuration inputs consistent with the requested benchmarking setup
batch_size = 4096
dim = 393216

def get_init_inputs():
    return []

def get_inputs():
    x = torch.rand(batch_size, dim, device='cuda', dtype=torch.float32)
    return [x]
