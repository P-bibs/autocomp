# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_010905/code_30.py
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

# -------------------------------------------------------------------------
# CUDA kernel – grid-stride loop, float4 vectorized, one block per row
# -------------------------------------------------------------------------
# Each block handles one row of the input. Since dim (393216) is large, 
# this ensures enough work per block to keep SMs saturated while 
# minimizing kernel launch overhead and maximizing memory throughput.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void tanh_kernel_vec4(const float* __restrict__ input,
                                 float* __restrict__ output,
                                 const int dim) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    
    // Pointer to the start of the current row
    const float* row_in = input + (size_t)row * dim;
    float* row_out = output + (size_t)row * dim;
    
    // Each thread processes elements with a stride of blockDim.x * 4
    // Using 4-element vectorization (float4) for coalesced memory access
    const int stride = blockDim.x * 4;
    
    for (int i = tid * 4; i < dim; i += stride) {
        if (i + 4 <= dim) {
            // Load 4 elements at once
            float4 in_vec = *reinterpret_cast<const float4*>(row_in + i);
            float4 out_vec;
            out_vec.x = tanhf(in_vec.x);
            out_vec.y = tanhf(in_vec.y);
            out_vec.z = tanhf(in_vec.z);
            out_vec.w = tanhf(in_vec.w);
            *reinterpret_cast<float4*>(row_out + i) = out_vec;
        } else {
            // Cleanup loop for remaining elements (if any, though dim here is multiple of 4)
            for (int j = i; j < dim && j < i + 4; ++j) {
                row_out[j] = tanhf(row_in[j]);
            }
        }
    }
}

void launch_tanh_kernel(const torch::Tensor& input, torch::Tensor& output) {
    const int batch = input.size(0);
    const int dim = input.size(1);
    
    // 256 threads is generally optimal for occupancy on compute capability 7.5+
    const int threads_per_block = 256;
    
    // One block per batch entry to maximize SM utilization
    tanh_kernel_vec4<<<batch, threads_per_block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        dim
    );
}
"""

# -------------------------------------------------------------------------
# C++ binding
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void launch_tanh_kernel(const torch::Tensor& input, torch::Tensor& output);

torch::Tensor custom_tanh(const torch::Tensor& input) {
    auto output = torch::empty_like(input);
    launch_tanh_kernel(input, output);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("custom_tanh", &custom_tanh, "Optimized vectorized tanh");
}
"""

# Compile the extension (Inline loading)
tanh_ext = load_inline(
    name='tanh_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x):
    """
    Applies the custom highly optimized tanh CUDA kernel.
    Input must be float32 on GPU.
    """
    return tanh_ext.custom_tanh(x)

# Configuration for benchmarking
batch_size = 4096
dim = 393216

def get_init_inputs():
    return []

def get_inputs():
    # Ensure tensor is contiguous and on GPU for coalesced access
    x = torch.rand(batch_size, dim, device='cuda', dtype=torch.float32)
    return [x]
