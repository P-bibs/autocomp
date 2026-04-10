# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_075452/code_2.py
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

template <typename scalar_t>
__global__ void cumsum_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int rows,
    const int cols) {
    
    // Each block handles one or more rows
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    if (row >= rows) return;
    
    // Shared memory for coalesced access and reduction
    extern __shared__ char shared_mem[];
    scalar_t* shared_data = reinterpret_cast<scalar_t*>(shared_mem);
    
    const scalar_t* row_input = input + row * cols;
    scalar_t* row_output = output + row * cols;
    
    scalar_t sum = 0;
    
    // Process the row in chunks
    for (int chunk_start = 0; chunk_start < cols; chunk_start += block_size) {
        int col = chunk_start + tid;
        
        // Load data into shared memory
        if (col < cols) {
            shared_data[tid] = row_input[col];
        } else {
            shared_data[tid] = 0;
        }
        
        __syncthreads();
        
        // Perform cumulative sum within the shared memory chunk
        if (tid == 0) {
            shared_data[0] += sum;
            for (int i = 1; i < min(block_size, cols - chunk_start); i++) {
                shared_data[i] += shared_data[i-1];
            }
            // Update the running sum for next chunk
            if (chunk_start + block_size < cols) {
                sum = shared_data[min(block_size, cols - chunk_start) - 1];
            }
        }
        
        __syncthreads();
        
        // Write results back to global memory
        if (col < cols) {
            row_output[col] = shared_data[tid];
        }
        
        __syncthreads();
    }
}

void fused_op_forward(
    const torch::Tensor input,
    torch::Tensor output,
    const int dim) {
    
    // Ensure we're on the right device
    at::cuda::CUDAGuard device_guard(input.device());
    
    const int rows = (dim == 1) ? input.size(0) : input.size(1);
    const int cols = (dim == 1) ? input.size(1) : input.size(0);
    
    const int threads_per_row = 256;
    const int rows_per_block = 4;
    const int threads_x = threads_per_row;
    const int threads_y = rows_per_block;
    
    const int blocks_x = (rows + rows_per_block - 1) / rows_per_block;
    
    dim3 block_size(threads_x, threads_y);
    dim3 grid_size(blocks_x);
    
    const int shared_mem_size = threads_per_row * sizeof(float);
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "cumsum_kernel", ([&] {
        cumsum_kernel<scalar_t><<<grid_size, block_size, shared_mem_size>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            rows,
            cols
        );
    }));
}
"""

# --- C++ Logic (Interface/Bindings) ---
cpp_source = r"""
#include <torch/extension.h>

// Forward declaration of the function in the .cu file
void fused_op_forward(const torch::Tensor input, torch::Tensor output, const int dim);

torch::Tensor fused_op(const torch::Tensor input, const int dim) {
    auto output = torch::empty_like(input);
    fused_op_forward(input, output, dim);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op, "Custom cumulative sum operation");
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
    return fused_ext.fused_op(x, dim)

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
    return [torch.rand(batch_size, *input_shape, device='cuda')]
