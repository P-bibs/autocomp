# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_231352/code_2.py
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define CUDA kernel
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void tanh_forward_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int numel
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < numel) {
        float val = x[idx];
        out[idx] = tanhf(val);
    }
}

void tanh_forward_cuda(
    torch::Tensor x,
    torch::Tensor out
) {
    int numel = x.numel();
    int block_size = 256;
    int grid_size = (numel + block_size - 1) / block_size;
    
    tanh_forward_kernel<<<grid_size, block_size>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        numel
    );
    
    cudaDeviceSynchronize();
}
"""

# Define C++ binding
cpp_source = r"""
#include <torch/extension.h>

void tanh_forward_cuda(
    torch::Tensor x,
    torch::Tensor out
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tanh_forward", &tanh_forward_cuda, "Fast tanh forward pass using CUDA");
}
"""

# Compile the extension with optimization flags
tanh_ext = load_inline(
    name='tanh_fast',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True,
    verbose=False
)

def functional_model(x):
    """
    Compute element-wise tanh using optimized CUDA kernel.
    
    Args:
        x: Input tensor of shape (batch_size, dim)
    
    Returns:
        Output tensor of same shape with tanh applied
    """
    # Ensure input is contiguous and on CUDA
    x = x.cuda().contiguous()
    
    # Allocate output tensor
    out = torch.empty_like(x)
    
    # Call the optimized CUDA kernel
    tanh_ext.tanh_forward(x, out)
    
    return out

batch_size = 4096
dim = 393216

def get_init_inputs():
    return []

def get_inputs():
    x = torch.rand(batch_size, dim)
    return [x]
