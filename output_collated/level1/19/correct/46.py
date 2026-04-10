# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_185153/code_21.py
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

# The grid-stride loop approach with float4 vectorization maximizes memory throughput.
# By using fixed grid dimensions, we maintain consistent occupation. 
# The remainder logic is simplified to ensure correctness for any input size.

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void relu_vec_kernel(const float* __restrict__ input, float* __restrict__ output, const size_t num_elements) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = blockDim.x * gridDim.x;
    
    // Each thread processes chunks of 4 floats (float4)
    // Grid-stride loop handles arbitrary sizes efficiently
    int i = tid * 4;
    for (; i + 3 < num_elements; i += total_threads * 4) {
        float4 in_vec = reinterpret_cast<const float4*>(&input[i])[0];
        float4 out_vec;
        
        out_vec.x = fmaxf(0.0f, in_vec.x);
        out_vec.y = fmaxf(0.0f, in_vec.y);
        out_vec.z = fmaxf(0.0f, in_vec.z);
        out_vec.w = fmaxf(0.0f, in_vec.w);
        
        reinterpret_cast<float4*>(&output[i])[0] = out_vec;
    }

    // Remainder handling: only the threads that have remaining elements need to process them
    // This is typically a very small number of elements at the end
    for (; i < num_elements; ++i) {
        output[i] = fmaxf(0.0f, input[i]);
    }
}

void launch_relu_kernel(const float* input, float* output, const size_t num_elements) {
    const int threads_per_block = 256;
    // 1024 blocks provides sufficient occupancy for the 2080Ti (which has 68 SMs)
    const int num_blocks = 1024;
    relu_vec_kernel<<<num_blocks, threads_per_block>>>(input, output, num_elements);
}
"""

cpp_source = r"""
#include <torch/extension.h>

void launch_relu_kernel(const float* input, float* output, const size_t num_elements);

void relu_forward(torch::Tensor input, torch::Tensor output) {
    launch_relu_kernel(input.data_ptr<float>(), output.data_ptr<float>(), input.numel());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("relu_forward", &relu_forward, "High performance ReLU using float4 and grid-stride loops");
}
"""

# Compile the extension inline
relu_ext = load_inline(
    name='relu_op_optimized',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x):
    """
    Optimized ReLU function that ensures input coherency and calls custom CUDA kernel.
    """
    # Ensure memory is contiguous to allow safe float4 reinterpret_cast
    if not x.is_contiguous():
        x = x.contiguous()
    
    output = torch.empty_like(x)
    relu_ext.relu_forward(x, output)
    return output

# Helper variables as requested
batch_size = 4096
dim = 393216

def get_init_inputs():
    return []

def get_inputs():
    x = torch.rand(batch_size, dim, device='cuda', dtype=torch.float32)
    return [x]
