# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_184018/code_21.py
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
# CUDA source (kernel + host function)
# Optimized: Split kernels to eliminate warp divergence, using float4
# and read-only cache loads for memory-bound throughput.
# -------------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Full-vector kernel: each thread processes exactly one float4 (4 elements)
// Optimized for throughput using float4 vector instructions and __ldg() for L1/Read-only caching.
__global__ void relu_vec_kernel(const float* __restrict__ input,
                                 float* __restrict__ output,
                                 size_t full_vectors) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < full_vectors) {
        // Load via read-only cache
        float4 in_vec = __ldg(reinterpret_cast<const float4*>(input) + idx);
        
        float4 out_vec;
        out_vec.x = fmaxf(in_vec.x, 0.0f);
        out_vec.y = fmaxf(in_vec.y, 0.0f);
        out_vec.z = fmaxf(in_vec.z, 0.0f);
        out_vec.w = fmaxf(in_vec.w, 0.0f);
        
        reinterpret_cast<float4*>(output)[idx] = out_vec;
    }
}

// Remainder kernel: handles 0-3 elements if n is not divisible by 4.
// Launched with minimal threads, avoiding divergence in the hot path.
__global__ void relu_rem_kernel(const float* __restrict__ input,
                                float* __restrict__ output,
                                size_t start,
                                size_t count) {
    size_t tid = threadIdx.x;
    if (tid < count) {
        size_t pos = start + tid;
        output[pos] = fmaxf(input[pos], 0.0f);
    }
}

void fused_op_forward(torch::Tensor input, torch::Tensor output) {
    size_t n = input.numel();
    if (n == 0) return;
    
    const int threads = 256;
    size_t full_vectors = n / 4;
    size_t remainder = n % 4;

    // Launch main vector kernel
    if (full_vectors > 0) {
        int blocks = static_cast<int>((full_vectors + threads - 1) / threads);
        relu_vec_kernel<<<blocks, threads>>>(
            input.data_ptr<float>(), output.data_ptr<float>(), full_vectors);
    }

    // Launch remainder kernel if necessary
    if (remainder > 0) {
        relu_rem_kernel<<<1, static_cast<int>(remainder)>>>(
            input.data_ptr<float>(), output.data_ptr<float>(),
            full_vectors * 4, remainder);
    }
}
"""

# -------------------------------------------------------------------------
# C++ binding
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>
void fused_op_forward(torch::Tensor input, torch::Tensor output);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Optimized Coalesced ReLU kernel");
}
"""

# Load the extension
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x: torch.Tensor) -> torch.Tensor:
    """
    High-performance ReLU implementation.
    Allocates output and dispatches custom CUDA kernels to perform the op.
    """
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
