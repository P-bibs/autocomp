# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_233710/code_30.py
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

# The grid-stride loop approach ensures optimal GPU occupancy and allows 
# the kernel to handle arbitrary sizes while maintaining memory coalescing
# through float4 vectorization.

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void tanh_kernel_vec4_grid(const float* __restrict__ input,
                                       float* __restrict__ output,
                                       int numel) {
    // Number of complete float4 vectors in the tensor
    int num_vecs = numel >> 2;
    // Stride = total number of threads in the grid
    int vec_stride = blockDim.x * gridDim.x;

    // Grid-stride loop: each thread processes multiple vectors
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < num_vecs;
         i += vec_stride) {
        
        float4 in_vec = reinterpret_cast<const float4*>(input)[i];
        float4 out_vec;
        out_vec.x = tanhf(in_vec.x);
        out_vec.y = tanhf(in_vec.y);
        out_vec.z = tanhf(in_vec.z);
        out_vec.w = tanhf(in_vec.w);
        reinterpret_cast<float4*>(output)[i] = out_vec;
    }

    // Cleanup: Handle the remaining elements if numel is not a multiple of 4
    // Only threads that would process the last "partial" block check the tail
    int remaining_start = num_vecs * 4;
    for (int i = remaining_start + (blockIdx.x * blockDim.x + threadIdx.x);
         i < numel;
         i += vec_stride) {
        output[i] = tanhf(input[i]);
    }
}

void launch_tanh_kernel(const torch::Tensor& input, torch::Tensor& output) {
    const int numel = input.numel();
    const int threads_per_block = 256;
    
    // Use occupancy-based or fixed reasonable grid size
    // 1024 blocks is generally sufficient to saturate modern GPUs
    const int num_vecs = numel >> 2;
    int grid_dim = (num_vecs + threads_per_block - 1) / threads_per_block;
    if (grid_dim > 1024) grid_dim = 1024;
    if (grid_dim == 0) grid_dim = 1;

    tanh_kernel_vec4_grid<<<grid_dim, threads_per_block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        numel
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void launch_tanh_kernel(const torch::Tensor& input, torch::Tensor& output);

torch::Tensor custom_tanh(const torch::Tensor& input) {
    auto output = torch::empty_like(input);
    launch_tanh_kernel(input, output);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("custom_tanh", &custom_tanh, "Optimized vectorized CUDA tanh");
}
"""

# Compile the extension
tanh_ext = load_inline(
    name='tanh_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x):
    """
    Applies the custom tanh kernel to the input tensor.
    The implementation is optimized using a grid-stride loop and float4
    memory access to maximize memory throughput on the RTX 2080Ti.
    """
    return tanh_ext.custom_tanh(x)

# ----------------------------------------------------------------------
# Interface required by the evaluation harness
# ----------------------------------------------------------------------
batch_size = 4096
dim = 393216

def get_init_inputs():
    return []

def get_inputs():
    # Keep inputs on GPU as per requirements
    x = torch.rand(batch_size, dim, device='cuda', dtype=torch.float32)
    return [x]
