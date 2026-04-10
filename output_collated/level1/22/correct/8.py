# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_231959/code_2.py
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

# Custom CUDA kernel for fused tanh operation
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void tanh_kernel(const float* __restrict__ input, float* __restrict__ output, int num_elements) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < num_elements; i += stride) {
        float x = input[i];
        // Using tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
        float exp_2x = expf(2.0f * x);
        output[i] = (exp_2x - 1.0f) / (exp_2x + 1.0f);
    }
}

__global__ void tanh_kernel_double(const double* __restrict__ input, double* __restrict__ output, int num_elements) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < num_elements; i += stride) {
        double x = input[i];
        // Using tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
        double exp_2x = exp(2.0 * x);
        output[i] = (exp_2x - 1.0) / (exp_2x + 1.0);
    }
}

void tanh_forward_float(int blocks, int threads, const torch::Tensor& input, torch::Tensor& output) {
    const float* input_ptr = input.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    int num_elements = input.numel();
    
    tanh_kernel<<<blocks, threads>>>(input_ptr, output_ptr, num_elements);
}

void tanh_forward_double(int blocks, int threads, const torch::Tensor& input, torch::Tensor& output) {
    const double* input_ptr = input.data_ptr<double>();
    double* output_ptr = output.data_ptr<double>();
    int num_elements = input.numel();
    
    tanh_kernel_double<<<blocks, threads>>>(input_ptr, output_ptr, num_elements);
}
"""

# C++ bindings
cpp_source = r"""
#include <torch/extension.h>

void tanh_forward_float(int blocks, int threads, const torch::Tensor& input, torch::Tensor& output);
void tanh_forward_double(int blocks, int threads, const torch::Tensor& input, torch::Tensor& output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tanh_op_float", &tanh_forward_float, "Custom Tanh Forward Pass (Float)");
    m.def("tanh_op_double", &tanh_forward_double, "Custom Tanh Forward Pass (Double)");
}
"""

# Compile the extension
tanh_ext = load_inline(
    name='tanh_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x):
    # Ensure tensor is on GPU
    if not x.is_cuda:
        x = x.cuda()
        
    output = torch.empty_like(x)
    num_elements = x.numel()
    
    # Use optimal block size (multiple of 32) and grid size
    threads_per_block = 512
    blocks_per_grid = min(65535, (num_elements + threads_per_block - 1) // threads_per_block)
    
    # Call appropriate kernel based on data type
    if x.dtype == torch.float32:
        tanh_ext.tanh_op_float(blocks_per_grid, threads_per_block, x, output)
    elif x.dtype == torch.float64:
        tanh_ext.tanh_op_double(blocks_per_grid, threads_per_block, x, output)
    else:
        # For other data types, cast to float32
        x_float = x.float()
        output_float = torch.empty_like(x_float)
        tanh_ext.tanh_op_float(blocks_per_grid, threads_per_block, x_float, output_float)
        output = output_float.to(x.dtype)
    
    return output

batch_size = 4096
dim = 393216

def get_init_inputs():
    return []  # No special initialization inputs needed

def get_inputs():
    x = torch.rand(batch_size, dim, dtype=torch.float32, device='cuda')
    return [x]
