# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_184018/code_14.py
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

# CUDA kernel with vectorized float4 loads for maximum memory bandwidth utilization
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector_types.h>

__global__ void relu_kernel_vectorized(const float* __restrict__ input, float* __restrict__ output, int64_t num_elements) {
    // Each thread processes 4 elements at a time using float4
    int64_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    int64_t stride = (int64_t)blockDim.x * gridDim.x * 4;
    
    for (int64_t i = idx; i < num_elements; i += stride) {
        if (i + 3 < num_elements) {
            float4 in_vec = reinterpret_cast<const float4*>(input + i)[0];
            float4 out_vec;
            out_vec.x = fmaxf(0.0f, in_vec.x);
            out_vec.y = fmaxf(0.0f, in_vec.y);
            out_vec.z = fmaxf(0.0f, in_vec.z);
            out_vec.w = fmaxf(0.0f, in_vec.w);
            reinterpret_cast<float4*>(output + i)[0] = out_vec;
        } else {
            // Handle remaining elements if num_elements is not a multiple of 4
            for (int64_t j = i; j < num_elements && j < i + 4; ++j) {
                output[j] = fmaxf(0.0f, input[j]);
            }
        }
    }
}

void launch_relu(const torch::Tensor& input, torch::Tensor& output) {
    const int64_t num_elements = input.numel();
    const int threads = 256;
    // Calculate blocks based on 4 elements per thread
    const int blocks = std::min((int64_t)65535, (num_elements / 4 + threads - 1) / threads);
    
    relu_kernel_vectorized<<<blocks, threads>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        num_elements
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void launch_relu(const torch::Tensor& input, torch::Tensor& output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_relu", &launch_relu, "Vectorized ReLU Forward");
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
    if not x.is_cuda:
        x = x.to('cuda', non_blocking=True)
    
    # Pre-allocate output
    output = torch.empty_like(x)
    
    # Call the vectorized kernel
    fused_ext.fused_relu(x, output)
    
    return output

# Inputs as required
batch_size = 4096
dim = 393216

def get_init_inputs():
    return []

def get_inputs():
    return [torch.rand(batch_size, dim)]
