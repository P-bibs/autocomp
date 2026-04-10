# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_213956/code_19.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['negative_slope']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['negative_slope']
REQUIRED_FLAT_STATE_NAMES = []


class ModelNew(nn.Module):
    """
    Simple model that performs a LeakyReLU activation.
    """

    def __init__(self, negative_slope: float=0.01):
        """
        Initializes the LeakyReLU module.

        Args:
            negative_slope (float, optional): The negative slope of the activation function. Defaults to 0.01.
        """
        super(ModelNew, self).__init__()
        self.negative_slope = negative_slope

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
    if 'negative_slope' in flat_state:
        state_kwargs['negative_slope'] = flat_state['negative_slope']
    else:
        state_kwargs['negative_slope'] = getattr(model, 'negative_slope')
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

# CUDA kernel with grid-stride loop and float4 vectorization
# This approach maximizes memory bandwidth by ensuring aligned float4 access
# and minimizes kernel launch overhead via grid-stride loops.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector_types.h>

__device__ __forceinline__ float leaky_relu_op(float val, float neg_slope) {
    return val > 0 ? val : val * neg_slope;
}

__global__ void leaky_relu_vec4_kernel(const float* __restrict__ input, float* __restrict__ output, float negative_slope, int numel) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    
    // Process 4 elements at a time per iteration for vectorized memory access
    int step = stride * 4;
    for (int idx = tid * 4; idx < numel; idx += step) {
        if (idx + 3 < numel) {
            float4 in_vec = reinterpret_cast<const float4*>(input)[idx / 4];
            float4 out_vec;
            
            out_vec.x = leaky_relu_op(in_vec.x, negative_slope);
            out_vec.y = leaky_relu_op(in_vec.y, negative_slope);
            out_vec.z = leaky_relu_op(in_vec.z, negative_slope);
            out_vec.w = leaky_relu_op(in_vec.w, negative_slope);
            
            reinterpret_cast<float4*>(output)[idx / 4] = out_vec;
        } else {
            // Sequential cleanup for tail elements
            for (int i = idx; i < numel; ++i) {
                output[i] = leaky_relu_op(input[i], negative_slope);
            }
        }
    }
}

void launch_leaky_relu(int numel, float* input, float* output, float negative_slope) {
    int block_size = 256;
    int num_vec_elements = numel / 4;
    // Aim for enough blocks to fill the GPU (SM count * warps), then use grid-stride
    int grid_size = 1024; 
    leaky_relu_vec4_kernel<<<grid_size, block_size>>>(input, output, negative_slope, numel);
}
"""

cpp_source = r"""
#include <torch/extension.h>

void launch_leaky_relu(int numel, float* input, float* output, float negative_slope);

void leaky_relu_forward(torch::Tensor input, torch::Tensor output, float negative_slope) {
    launch_leaky_relu(input.numel(), input.data_ptr<float>(), output.data_ptr<float>(), negative_slope);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("leaky_relu_forward", &leaky_relu_forward, "Vectorized Leaky ReLU forward with grid-stride loop");
}
"""

# Compile the extension
leaky_relu_ext = load_inline(
    name='leaky_relu_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, negative_slope):
    """
    Optimized version of Leaky ReLU execution using custom CUDA kernel.
    """
    if not x.is_contiguous():
        x = x.contiguous()
    
    output = torch.empty_like(x)
    leaky_relu_ext.leaky_relu_forward(x, output, float(negative_slope))
    return output

batch_size = 4096
dim = 393216

def get_init_inputs():
    return []

def get_inputs():
    x = torch.rand(batch_size, dim, device='cuda', dtype=torch.float32)
    return [x]
