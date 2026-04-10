# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_075452/code_8.py
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

# CUDA kernel for custom cumsum implementation
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

__device__ inline float warp_scan(float val) {
    for (int offset = 1; offset < 32; offset *= 2) {
        val += __shfl_up_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__global__ void cumsum_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int rows,
    const int cols
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int bdim = blockDim.x;
    
    if (row >= rows) return;
    
    const float* input_row = input + row * cols;
    float* output_row = output + row * cols;
    
    // Shared memory for block-level scan
    extern __shared__ float shared_data[];
    float* sdata = shared_data;
    
    float sum = 0.0f;
    for (int col = tid; col < cols; col += bdim) {
        sum += input_row[col];
        sdata[tid] = sum;
        __syncthreads();
        
        // Perform block-level inclusive scan
        for (int stride = 1; stride < bdim; stride *= 2) {
            float temp = 0.0f;
            if (tid >= stride) {
                temp = sdata[tid - stride];
            }
            __syncthreads();
            if (tid >= stride) {
                sdata[tid] += temp;
            }
            __syncthreads();
        }
        
        output_row[col] = sdata[tid];
        sum = sdata[bdim - 1]; // Last element contains the block sum
        __syncthreads();
    }
}

void cumsum_cuda(
    const at::Tensor& input,
    at::Tensor& output,
    const int dim
) {
    const int rows = input.size(0);
    const int cols = input.size(1);
    
    // Set the CUDA device guard to ensure we're on the correct device
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    
    const int threads = 256;
    const int blocks = rows;
    const int shared_mem_size = threads * sizeof(float);
    
    cumsum_kernel<<<blocks, threads, shared_mem_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        rows,
        cols
    );
    
    cudaDeviceSynchronize();
}
"""

# C++ interface for the CUDA kernel
cpp_source = r"""
#include <torch/extension.h>

void cumsum_cuda(
    const at::Tensor& input,
    at::Tensor& output,
    const int dim
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cumsum_cuda", &cumsum_cuda, "Custom CUDA implementation of cumsum");
}
"""

# Compile the extension
cumsum_ext = load_inline(
    name='cumsum_ext',
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
    # Create output tensor with same shape and device as input
    output = torch.empty_like(x)
    # Call our custom CUDA cumsum implementation
    cumsum_ext.cumsum_cuda(x, output, dim)
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
