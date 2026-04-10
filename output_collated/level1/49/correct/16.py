# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_154447/code_0.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['dim']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['dim']
REQUIRED_FLAT_STATE_NAMES = []


class ModelNew(nn.Module):
    """
    Simple model that performs Max reduction over a specific dimension.
    """

    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): The dimension to reduce over.
        """
        super(ModelNew, self).__init__()
        self.dim = dim

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
    if 'dim' in flat_state:
        state_kwargs['dim'] = flat_state['dim']
    else:
        state_kwargs['dim'] = getattr(model, 'dim')
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

# Custom CUDA kernel for max reduction
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

template <typename scalar_t>
__global__ void max_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int dim1,
    const int dim2,
    const int reduce_dim
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * dim1 * dim2;
    
    if (reduce_dim == 2) {
        // Reduce along the last dimension (dim2)
        const int output_stride = dim1;
        const int input_stride = dim2;
        const int total_reductions = batch_size * dim1;
        
        if (tid < total_reductions) {
            const int batch_idx = tid / dim1;
            const int dim1_idx = tid % dim1;
            const int base_idx = batch_idx * dim1 * dim2 + dim1_idx * dim2;
            
            scalar_t max_val = input[base_idx];
            for (int i = 1; i < dim2; i++) {
                scalar_t val = input[base_idx + i];
                max_val = fmaxf(max_val, val);
            }
            output[batch_idx * output_stride + dim1_idx] = max_val;
        }
    } else if (reduce_dim == 1) {
        // Reduce along dimension 1 (dim1)
        const int total_reductions = batch_size * dim2;
        
        if (tid < total_reductions) {
            const int batch_idx = tid / dim2;
            const int dim2_idx = tid % dim2;
            const int base_idx = batch_idx * dim1 * dim2 + dim2_idx;
            
            scalar_t max_val = input[base_idx];
            for (int i = 1; i < dim1; i++) {
                scalar_t val = input[base_idx + i * dim2];
                max_val = fmaxf(max_val, val);
            }
            output[batch_idx * dim2 + dim2_idx] = max_val;
        }
    }
}

void max_reduce_forward(
    const torch::Tensor& input,
    torch::Tensor& output,
    const int reduce_dim
) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    
    const int batch_size = input.size(0);
    const int dim1 = input.size(1);
    const int dim2 = input.size(2);
    
    const int threads_per_block = 256;
    int blocks;
    
    if (reduce_dim == 2) {
        blocks = (batch_size * dim1 + threads_per_block - 1) / threads_per_block;
    } else if (reduce_dim == 1) {
        blocks = (batch_size * dim2 + threads_per_block - 1) / threads_per_block;
    } else {
        return; // Unsupported dimension
    }
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_reduce_forward", ([&] {
        max_reduce_kernel<scalar_t><<<blocks, threads_per_block>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            dim1,
            dim2,
            reduce_dim
        );
    }));
    
    cudaDeviceSynchronize();
}
"""

# C++ interface
cpp_source = r"""
#include <torch/extension.h>

void max_reduce_forward(
    const torch::Tensor& input,
    torch::Tensor& output,
    const int reduce_dim
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("max_reduce", &max_reduce_forward, "Max reduction along specified dimension");
}
"""

# Compile the extension
max_reduce_ext = load_inline(
    name='max_reduce_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(
    x,
    *,
    dim,
):
    # Create output tensor with correct shape
    if dim == 2:
        output_shape = (x.size(0), x.size(1))
    elif dim == 1:
        output_shape = (x.size(0), x.size(2))
    else:
        raise ValueError("Only dim=1 or dim=2 supported")
    
    output = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    max_reduce_ext.max_reduce(x, output, dim)
    return output

batch_size = 128
dim1 = 4096
dim2 = 4095

def get_init_inputs():
    return [1] # Example, change to desired dimension

def get_inputs():
    x = torch.rand(batch_size, dim1, dim2)
    return [x]
