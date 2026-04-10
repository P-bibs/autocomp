# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_081907/code_8.py
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

# Custom CUDA kernel for parallel prefix sum
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

__device__ void warp_scan(float* data, int tid) {
    // Perform warp-level inclusive scan using shfl operations
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        float temp = __shfl_up_sync(0xffffffff, data[tid], offset);
        if (tid >= offset) {
            data[tid] += temp;
        }
    }
}

__device__ void block_scan(float* data, int tid, int block_size) {
    // Shared memory for block-level operations
    __shared__ float shared_data[1024];
    __shared__ float block_sums[32]; // Up to 32 warps per block
    
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int num_warps = (block_size + 31) / 32;
    
    // Warp-level scan
    warp_scan(data, tid);
    
    // Store warp sums
    if (lane_id == 31) {
        block_sums[warp_id] = data[tid];
    }
    __syncthreads();
    
    // Scan the warp sums (only first warp does this)
    if (warp_id == 0 && lane_id < num_warps) {
        warp_scan(block_sums, lane_id);
    }
    __syncthreads();
    
    // Add warp sums to each element
    if (warp_id > 0) {
        data[tid] += block_sums[warp_id - 1];
    }
}

__global__ void cumsum_kernel(
    const float* input,
    float* output,
    int batch_size,
    int elements_per_row
) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    if (batch_idx >= batch_size) return;
    
    // Shared memory for the current row
    extern __shared__ float shared_data[];
    
    // Load data into shared memory
    for (int i = tid; i < elements_per_row; i += block_size) {
        shared_data[i] = input[batch_idx * elements_per_row + i];
    }
    __syncthreads();
    
    // Perform block-level scan on shared memory
    block_scan(shared_data, tid, block_size);
    __syncthreads();
    
    // Write back to global memory
    for (int i = tid; i < elements_per_row; i += block_size) {
        output[batch_idx * elements_per_row + i] = shared_data[i];
    }
}

void cumsum_cuda_kernel(
    const at::Tensor& input,
    at::Tensor& output,
    int dim
) {
    auto input_ptr = input.data_ptr<float>();
    auto output_ptr = output.data_ptr<float>();
    
    int batch_size = input.size(0);
    int elements_per_row = input.size(1);
    
    // Launch configuration
    int threads_per_block = min(1024, elements_per_row);
    threads_per_block = (threads_per_block / 32) * 32; // Multiple of warp size
    if (threads_per_block == 0) threads_per_block = 32;
    
    int shared_mem_size = elements_per_row * sizeof(float);
    
    cumsum_kernel<<<batch_size, threads_per_block, shared_mem_size>>>(
        input_ptr,
        output_ptr,
        batch_size,
        elements_per_row
    );
}
"""

# C++ interface/bindings
cpp_source = r"""
#include <torch/extension.h>

void cumsum_cuda_kernel(
    const at::Tensor& input,
    at::Tensor& output,
    int dim
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cumsum_cuda", &cumsum_cuda_kernel, "Custom CUDA CumSum implementation");
}
"""

# Compile the extension with optimization flags
custom_cumsum_ext = load_inline(
    name='custom_cumsum',
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
    # Create output tensor with same shape and dtype as input
    output = torch.empty_like(x)
    # Call custom CUDA kernel
    custom_cumsum_ext.cumsum_cuda(x, output, dim)
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
