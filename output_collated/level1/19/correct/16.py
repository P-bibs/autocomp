# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_184018/code_16.py
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

# --- CUDA Kernel ---
# We use __ldg for read-only coalesced global access.
# The kernel is manually vectorized for efficiency, processing 1 element per thread.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void relu_kernel(const float* __restrict__ x, float* __restrict__ y, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // __fmaxf is the intrinsic for fast max, effectively utilizing --use_fast_math
        y[idx] = fmaxf(0.0f, __ldg(&x[idx]));
    }
}

void fused_op_forward(torch::Tensor x, torch::Tensor y) {
    size_t n = x.numel();
    const int threads = 512;
    // Ensure we have enough blocks to cover all elements
    const int blocks = (n + threads - 1) / threads;
    relu_kernel<<<blocks, threads>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
}
"""

# --- C++ Binding ---
cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(torch::Tensor x, torch::Tensor y);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fast ReLU implementation");
}
"""

# Compile the extension
# Using -O3 and --use_fast_math for maximum throughput on RTX 2080Ti
fused_ext = load_inline(
    name='fused_relu_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x):
    """
    Optimized ReLU functional interface.
    Allocates output on the same device as input and invokes the custom CUDA kernel.
    """
    y = torch.empty_like(x)
    fused_ext.fused_op(x, y)
    return y

# Setup logic matching requirements
batch_size = 4096
dim = 393216

def get_init_inputs():
    return []

def get_inputs():
    # Ensure inputs are on CUDA as requested
    x = torch.rand(batch_size, dim, device='cuda', dtype=torch.float32)
    return [x]
