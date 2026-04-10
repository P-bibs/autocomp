# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_074424/code_8.py
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

# CUDA kernel for cumsum operation
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void cumsum_dim1_kernel(
    const float* input,
    float* output,
    int rows,
    int cols
) {
    int row = blockIdx.x;
    if (row >= rows) return;
    
    const float* row_input = input + row * cols;
    float* row_output = output + row * cols;
    
    // Initialize first element
    if (threadIdx.x == 0) {
        row_output[0] = row_input[0];
    }
    __syncthreads();
    
    // Sequential cumsum within each row
    for (int i = threadIdx.x + 1; i < cols; i += blockDim.x) {
        row_output[i] = row_input[i] + row_output[i-1];
        __syncthreads();
    }
}

__global__ void cumsum_dim0_kernel(
    const float* input,
    float* output,
    int rows,
    int cols
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= cols) return;
    
    // Initialize first element
    output[col] = input[col];
    
    // Sequential cumsum along column
    for (int i = 1; i < rows; i++) {
        output[i * cols + col] = input[i * cols + col] + output[(i-1) * cols + col];
    }
}

void cumsum_forward(
    torch::Tensor input,
    torch::Tensor output,
    int dim
) {
    int rows = input.size(0);
    int cols = input.size(1);
    
    float* input_ptr = input.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    
    if (dim == 1) {
        // Launch one block per row
        dim3 grid(rows);
        dim3 block(min(cols, 1024));
        cumsum_dim1_kernel<<<grid, block>>>(input_ptr, output_ptr, rows, cols);
    } else if (dim == 0) {
        // Launch one thread per column
        dim3 grid((cols + 1023) / 1024);
        dim3 block(1024);
        cumsum_dim0_kernel<<<grid, block>>>(input_ptr, output_ptr, rows, cols);
    }
    
    cudaDeviceSynchronize();
}
"""

# C++ bindings
cpp_source = r"""
#include <torch/extension.h>

void cumsum_forward(
    torch::Tensor input,
    torch::Tensor output,
    int dim
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cumsum_fwd", &cumsum_forward, "Cumsum CUDA forward pass");
}
"""

# Compile the extension
cumsum_ext = load_inline(
    name='cumsum_cuda',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, dim):
    """
    Optimized cumsum using custom CUDA kernel.
    
    Args:
        x: Input tensor
        dim: Dimension along which to compute cumsum
    
    Returns:
        Cumulative sum tensor
    """
    # Ensure input is contiguous and on CUDA
    x = x.contiguous().cuda()
    output = torch.empty_like(x)
    
    cumsum_ext.cumsum_fwd(x, output, dim)
    
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
