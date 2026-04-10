# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_073025/code_0.py
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

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

__global__ void cumsum_kernel(float* input, float* output, int rows, int cols) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    // Shared memory for the block
    extern __shared__ float shared[];
    
    if (row < rows) {
        // Process this row in chunks that fit in shared memory
        for (int chunk_start = 0; chunk_start < cols; chunk_start += block_size) {
            int chunk_size = min(block_size, cols - chunk_start);
            
            // Load data into shared memory with coalesced access
            if (tid < chunk_size) {
                shared[tid] = input[row * cols + chunk_start + tid];
            }
            __syncthreads();
            
            // Perform inclusive scan in shared memory using Kogge-Stone algorithm
            for (int stride = 1; stride < chunk_size; stride *= 2) {
                float temp = 0;
                if (tid >= stride && tid < chunk_size) {
                    temp = shared[tid - stride];
                }
                __syncthreads();
                if (tid >= stride && tid < chunk_size) {
                    shared[tid] += temp;
                }
                __syncthreads();
            }
            
            // Write result back to global memory with coalesced access
            if (tid < chunk_size) {
                output[row * cols + chunk_start + tid] = shared[tid];
            }
            __syncthreads();
        }
    }
}

void fused_op_forward(int64_t rows, int64_t cols, torch::Tensor input, torch::Tensor output) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    
    // Use optimal block size for memory coalescing
    int threads_per_block = 512;
    int blocks = rows;
    
    // Calculate shared memory size
    size_t shared_mem_size = threads_per_block * sizeof(float);
    
    cumsum_kernel<<<blocks, threads_per_block, shared_mem_size>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        rows, 
        cols
    );
}
"""

# --- C++ Logic (Interface/Bindings) ---
cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(int64_t rows, int64_t cols, torch::Tensor input, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused cumsum operation");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_op',
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
    # Handle different dimensions by transposing if necessary
    if dim != -1 and dim != x.dim() - 1:
        # For non-last dimension, we need to transpose
        permute_dims = list(range(x.dim()))
        permute_dims[dim], permute_dims[-1] = permute_dims[-1], permute_dims[dim]
        x = x.permute(*permute_dims)
        result = functional_model(x, dim=-1)
        return result.permute(*permute_dims)
    
    # Ensure input is contiguous and float32 for our kernel
    original_dtype = x.dtype
    if x.dtype != torch.float32:
        x = x.float()
    
    x = x.contiguous()
    output = torch.empty_like(x)
    
    # Handle different input shapes
    if x.dim() == 1:
        rows = 1
        cols = x.size(0)
        x_2d = x.unsqueeze(0)
        output_2d = output.unsqueeze(0)
    elif x.dim() == 2:
        rows = x.size(0)
        cols = x.size(1)
        x_2d = x
        output_2d = output
    else:
        # Reshape to 2D
        original_shape = x.shape
        x_2d = x.view(-1, x.size(-1))
        output_2d = output.view(-1, x.size(-1))
        rows = x_2d.size(0)
        cols = x_2d.size(1)
    
    # Call custom CUDA kernel
    fused_ext.fused_op(rows, cols, x_2d, output_2d)
    
    # Reshape back if needed and convert dtype
    result = output.view_as(x)
    if result.dtype != original_dtype:
        result = result.to(original_dtype)
        
    return result

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
