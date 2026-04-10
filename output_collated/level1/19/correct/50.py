# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_185153/code_29.py
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

# ----------------------------------------------------------------------
# Optimized CUDA kernel: Vectorized float4 with Grid-Stride Loop
# ----------------------------------------------------------------------
# Optimization: Replaced one-thread-per-4-elements with a grid-stride loop.
# This prevents excessive kernel launch overhead and matches the thread grid
# to the GPU's hardware capacity, significantly improving performance.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector_types.h>

__global__ void relu_vec_kernel(const float* __restrict__ input,
                                float* __restrict__ output,
                                const size_t num_elements) {
    // Grid-stride: each thread processes multiple 4-element chunks
    // 'stride' represents the total number of floats covered by one full grid invocation
    const size_t thread_stride = (size_t)blockDim.x * gridDim.x;
    const size_t element_stride = thread_stride * 4;
    
    size_t idx = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 4;
    
    for (; idx < num_elements; idx += element_stride) {
        // Use float4 for coalesced, vectorized memory access
        if (idx + 3 < num_elements) {
            float4 in_vec = reinterpret_cast<const float4*>(input)[idx / 4];
            float4 out_vec;
            
            out_vec.x = fmaxf(0.0f, in_vec.x);
            out_vec.y = fmaxf(0.0f, in_vec.y);
            out_vec.z = fmaxf(0.0f, in_vec.z);
            out_vec.w = fmaxf(0.0f, in_vec.w);
            
            reinterpret_cast<float4*>(output)[idx / 4] = out_vec;
        } else {
            // Handle remainder for non-multiple-of-4 sizes
            for (size_t i = idx; i < num_elements; ++i) {
                output[i] = fmaxf(0.0f, input[i]);
            }
        }
    }
}

void launch_relu_kernel(const float* input, float* output, const size_t num_elements) {
    const int threads_per_block = 256;
    // Using a fixed, hardware-friendly grid size
    const int num_blocks = 1024;
    relu_vec_kernel<<<num_blocks, threads_per_block>>>(input, output, num_elements);
}
"""

# ----------------------------------------------------------------------
# C++ binding (PyBind11)
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void launch_relu_kernel(const float* input, float* output, const size_t num_elements);

void relu_forward(torch::Tensor input, torch::Tensor output) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(output.is_cuda(), "Output must be a CUDA tensor");
    launch_relu_kernel(input.data_ptr<float>(), output.data_ptr<float>(), input.numel());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("relu_forward", &relu_forward, "Vectorized ReLU forward pass with grid-stride loop");
}
"""

# Compile the inline CUDA extension
relu_ext = load_inline(
    name='relu_op_optimized',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x):
    """
    Optimized ReLU function using a custom CUDA kernel with grid-stride loops
    and vectorized 128-bit (float4) memory access.
    """
    if not x.is_contiguous():
        x = x.contiguous()
    
    output = torch.empty_like(x)
    relu_ext.relu_forward(x, output)
    return output
