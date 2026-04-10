# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_231352/code_6.py
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

# Define the CUDA kernel for element-wise Tanh
# We use __launch_bounds__ and vectorized inputs if possible, 
# but for a memory-bound tanh, standard 256-thread blocks are highly effective.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void tanh_kernel(const float* __restrict__ x, float* __restrict__ out, int numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        // tanhf is the standard fast intrinsic in CUDA for float tanh
        out[idx] = tanhf(x[idx]);
    }
}

void tanh_cuda_launch(torch::Tensor x, torch::Tensor out) {
    const int numel = x.numel();
    const int block_size = 256;
    const int grid_size = (numel + block_size - 1) / block_size;
    
    tanh_kernel<<<grid_size, block_size>>>(
        x.data_ptr<float>(), 
        out.data_ptr<float>(), 
        numel
    );
}
"""

# Define C++ binding
cpp_source = r"""
#include <torch/extension.h>

void tanh_cuda_launch(torch::Tensor x, torch::Tensor out);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tanh_forward", &tanh_cuda_launch, "Optimized Tanh CUDA Kernel");
}
"""

# Compile the extension
# Using -O3 and --use_fast_math for maximum throughput on the RTX 2080Ti
tanh_ext = load_inline(
    name='tanh_optimized',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True,
    verbose=False
)

def functional_model(x):
    """
    Optimized implementation of element-wise tanh.
    We ensure data is on GPU and use a custom kernel to avoid 
    ATen overhead and maximize memory bandwidth.
    """
    # Ensure tensor is on GPU and is contiguous for coalesced access
    x = x.cuda().contiguous()
    out = torch.empty_like(x)
    
    # Launch kernel
    tanh_ext.tanh_forward(x, out)
    
    return out

# Constants provided by the task
batch_size = 4096
dim = 393216

def get_init_inputs():
    return []

def get_inputs():
    return [torch.rand(batch_size, dim)]
