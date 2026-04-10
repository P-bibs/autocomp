# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_184018/code_19.py
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

# CUDA Kernel for vectorized ReLU
# We process 4 floats per thread using float4 vector loads/stores.
# This maximizes memory throughput by coalescing 128-bit transactions.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector_types.h>

__global__ void relu_vec4_kernel(const float* __restrict__ input, float* __restrict__ output, size_t num_elements) {
    size_t idx = (size_t)(blockIdx.x * blockDim.x + threadIdx.x) * 4;
    
    if (idx + 3 < num_elements) {
        float4 val = reinterpret_cast<const float4*>(input)[idx / 4];
        
        val.x = fmaxf(0.0f, val.x);
        val.y = fmaxf(0.0f, val.y);
        val.z = fmaxf(0.0f, val.z);
        val.w = fmaxf(0.0f, val.w);
        
        reinterpret_cast<float4*>(output)[idx / 4] = val;
    } else {
        // Handle remainder for non-aligned sizes
        for (size_t i = idx; i < num_elements && i < idx + 4; ++i) {
            output[i] = fmaxf(0.0f, input[i]);
        }
    }
}

void relu_vec4_launch(torch::Tensor input, torch::Tensor output) {
    size_t num_elements = input.numel();
    size_t num_vec_elements = num_elements / 4;
    
    const int threads = 256;
    const int blocks = (num_vec_elements + threads - 1) / threads;
    
    relu_vec4_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        num_elements
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void relu_vec4_launch(torch::Tensor input, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("relu_vec4", &relu_vec4_launch, "Vectorized ReLU kernel");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_relu',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x):
    """
    Optimized ReLU using vectorized float4 CUDA kernel.
    """
    output = torch.empty_like(x)
    fused_ext.relu_vec4(x, output)
    return output

# --- Evaluation setup constants ---
batch_size = 4096
dim = 393216

def get_init_inputs():
    return []

def get_inputs():
    return [torch.rand(batch_size, dim, device='cuda')]
