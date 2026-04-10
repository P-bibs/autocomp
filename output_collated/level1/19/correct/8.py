# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_184018/code_5.py
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

# Optimized CUDA kernel with flexible grid/block calculation
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void relu_kernel(const float* __restrict__ input, float* __restrict__ output, int64_t num_elements) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

void relu_forward_cuda(const torch::Tensor& input, torch::Tensor& output) {
    const int64_t num_elements = input.numel();
    const int threads = 256; // Multiple of warp size for better occupancy
    const int blocks = (num_elements + threads - 1) / threads;
    
    relu_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        num_elements
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void relu_forward_cuda(const torch::Tensor& input, torch::Tensor& output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_relu", &relu_forward_cuda, "Optimized C++ ReLU Forward");
}
"""

fused_ext = load_inline(
    name='fused_relu_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x):
    # Ensure input is on device
    if not x.is_cuda:
        x = x.to('cuda', non_blocking=True)
    
    output = torch.empty_like(x)
    
    # Execute the kernel with dynamically scaled block/grid distribution
    fused_ext.fused_relu(x, output)
    
    return output

batch_size = 4096
dim = 393216

def get_init_inputs():
    return []

def get_inputs():
    # Return inputs on CPU to match original behavior, 
    # the function converts them internally
    x = torch.rand(batch_size, dim)
    return [x]
