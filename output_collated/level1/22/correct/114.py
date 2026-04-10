# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_004356/code_30.py
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

# ----------------------------------------------------------------------
# CUDA kernel – grid-stride loop + float4 vectorisation
# ----------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdint>

// Optimized CUDA kernel using grid-stride loops and float4 vectorization
__global__ void tanh_kernel_vec4_grid(const float* __restrict__ input,
                                      float*       __restrict__ output,
                                      const int64_t numel) {
    // We process 4 floats at a time via float4
    const int vec_size = 4;
    // Calculate global index in terms of floats
    int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    
    // Each thread starts at (tid * 4) and jumps by (stride * 4)
    for (int64_t i = tid * vec_size; i < numel; i += stride * vec_size) {
        // If we have a full float4 vector remaining, perform vectorized op
        if (i + vec_size <= numel) {
            float4 in_vec = reinterpret_cast<const float4*>(input + i)[0];
            float4 out_vec;
            out_vec.x = tanhf(in_vec.x);
            out_vec.y = tanhf(in_vec.y);
            out_vec.z = tanhf(in_vec.z);
            out_vec.w = tanhf(in_vec.w);
            reinterpret_cast<float4*>(output + i)[0] = out_vec;
        } else {
            // Cleanup: individual elements for the tail of the tensor
            for (int j = 0; j < vec_size && (i + j) < numel; ++j) {
                output[i + j] = tanhf(input[i + j]);
            }
        }
    }
}

void launch_tanh_kernel(const torch::Tensor& input, torch::Tensor& output) {
    const int64_t numel = input.numel();
    const int threads_per_block = 256;
    // Use 2048 blocks to utilize the GPU occupancy while amortizing launch overhead
    const int blocks = 2048;

    tanh_kernel_vec4_grid<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        numel
    );
}
"""

# ----------------------------------------------------------------------
# C++ wrapper – pybind11 binding
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void launch_tanh_kernel(const torch::Tensor& input, torch::Tensor& output);

torch::Tensor custom_tanh(const torch::Tensor& input) {
    auto output = torch::empty_like(input);
    launch_tanh_kernel(input, output);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("custom_tanh", &custom_tanh, "Vectorized CUDA tanh with grid-stride loops");
}
"""

# ----------------------------------------------------------------------
# Build the inline extension
# ----------------------------------------------------------------------
tanh_ext = load_inline(
    name='tanh_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# ----------------------------------------------------------------------
# Functional interface required by the evaluation harness
# ----------------------------------------------------------------------
def functional_model(x):
    """Apply the custom optimized vectorized tanh."""
    return tanh_ext.custom_tanh(x)

# Parameters needed for the evaluation harness
batch_size = 4096
dim = 393216

def get_init_inputs():
    """No persistent state needed."""
    return []

def get_inputs():
    """Generate a random input tensor on the GPU."""
    # Ensure input is contiguous for optimal memory access
    x = torch.rand(batch_size, dim, device='cuda', dtype=torch.float32)
    return [x.contiguous()]
