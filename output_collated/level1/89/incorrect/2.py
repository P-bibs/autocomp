# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_073025/code_2.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['dim']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['dim']
REQUIRED_FLAT_STATE_NAMES = []


class ModelNew(nn.Module):
    """
    A simple model that performs a cumulative sum (prefix sum) operation along a specified dimension.

    Parameters:
        dim (int): The dimension along which to perform the scan operation.
    """

    def __init__(self, dim):
        """
        Initialize the Scan model.

        Args:
            dim (int): The dimension along which to perform the cumulative sum.
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

# CUDA kernel for optimized cumsum
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

template<typename scalar_t>
__global__ void cumsum_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int batch_size,
    int dim_size
) {
    // Each block processes one row (along the cumsum dimension)
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    if (row >= batch_size) return;
    
    // Shared memory for block-level parallel scan
    extern __shared__ char shared_mem[];
    scalar_t* shared_data = reinterpret_cast<scalar_t*>(shared_mem);
    
    scalar_t sum = 0;
    // Process elements in chunks to fit in shared memory
    for (int base_idx = 0; base_idx < dim_size; base_idx += block_size) {
        int idx = base_idx + tid;
        if (idx < dim_size) {
            shared_data[tid] = input[row * dim_size + idx];
        } else {
            shared_data[tid] = 0;
        }
        __syncthreads();
        
        // Perform block-level parallel prefix sum (inclusive scan)
        // Up-sweep phase
        for (int stride = 1; stride < block_size; stride *= 2) {
            int index = (tid + 1) * stride * 2 - 1;
            if (index < block_size) {
                shared_data[index] += shared_data[index - stride];
            }
            __syncthreads();
        }
        
        // Down-sweep phase
        if (tid == 0) {
            shared_data[block_size - 1] = 0;
        }
        __syncthreads();
        
        for (int stride = block_size / 2; stride > 0; stride /= 2) {
            int index = (tid + 1) * stride * 2 - 1;
            if (index < block_size) {
                scalar_t temp = shared_data[index];
                shared_data[index] += shared_data[index - stride];
                shared_data[index - stride] = temp;
            }
            __syncthreads();
        }
        
        // Add the running sum from previous chunks
        for (int i = 0; i < block_size && base_idx + i < dim_size; i++) {
            output[row * dim_size + base_idx + i] = shared_data[i] + sum;
        }
        
        // Update the running sum
        if (base_idx + block_size - 1 < dim_size) {
            sum += shared_data[block_size - 1] + input[row * dim_size + base_idx + block_size - 1];
        }
        __syncthreads();
    }
}

void cumsum_forward(
    torch::Tensor input,
    torch::Tensor output,
    int dim
) {
    int batch_size = input.size(0);
    int dim_size = input.size(1);
    
    // Use a block size that's a power of 2 and fits in shared memory
    const int threads_per_block = 512;
    const int blocks = batch_size;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "cumsum_kernel", ([&] {
        cumsum_kernel<scalar_t><<<blocks, threads_per_block, threads_per_block * sizeof(scalar_t)>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            dim_size
        );
    }));
    
    cudaDeviceSynchronize();
}
"""

# C++ binding
cpp_source = r"""
#include <torch/extension.h>

void cumsum_forward(
    torch::Tensor input,
    torch::Tensor output,
    int dim
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cumsum_fused", &cumsum_forward, "Fused cumsum operation");
}
"""

# Compile the extension with aggressive optimizations
cumsum_ext = load_inline(
    name='cumsum_fused',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, dim):
    """
    Optimized cumsum using custom CUDA kernel compiled with -O3 and --use_fast_math
    """
    output = torch.empty_like(x)
    cumsum_ext.cumsum_fused(x, output, dim)
    return output

batch_size = 32768
input_shape = (32768,)
dim = 1

def get_init_inputs():
    """
    Returns the initialization parameters for the Scan model.

    Returns:
        list: A list containing the `dim` parameter for model initialization.
    """
    return [dim]

def get_inputs():
    """
    Generates random inputs for testing the Scan model.

    Returns:
        list: A list containing a single randomly generated tensor with shape 
              (batch_size, *input_shape).
    """
    return [torch.rand(batch_size, *input_shape)]
