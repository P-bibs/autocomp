# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_080721/code_8.py
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

# CUDA kernel source code
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32

__device__ __forceinline__ float warp_scan(float val, int lane_id) {
    #pragma unroll
    for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
        float temp = __shfl_up_sync(0xFFFFFFFF, val, offset);
        if (lane_id >= offset) {
            val += temp;
        }
    }
    return val;
}

__global__ void inclusive_scan_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows,
    int cols)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    
    if (row >= rows) return;
    
    extern __shared__ float shared_data[];
    float* warp_sums = shared_data + blockDim.x;
    
    // Load data from global memory to shared memory
    float val = 0.0f;
    if (tid < cols) {
        val = input[row * cols + tid];
    }
    shared_data[tid] = val;
    __syncthreads();
    
    // Warp-level scans
    float warp_result = warp_scan(val, lane_id);
    shared_data[tid] = warp_result;
    
    // Store warp sums
    if (lane_id == WARP_SIZE - 1) {
        warp_sums[warp_id] = warp_result;
    }
    __syncthreads();
    
    // Scan the warp sums (only first warp does this)
    if (warp_id == 0 && lane_id < (blockDim.x + WARP_SIZE - 1) / WARP_SIZE) {
        float warp_sum = warp_sums[lane_id];
        warp_sum = warp_scan(warp_sum, lane_id);
        warp_sums[lane_id] = warp_sum;
    }
    __syncthreads();
    
    // Add warp sums back to elements
    if (tid < cols) {
        float warp_correction = 0.0f;
        if (warp_id > 0) {
            warp_correction = warp_sums[warp_id - 1];
        }
        output[row * cols + tid] = warp_result + warp_correction;
    }
}

void inclusive_scan_forward(
    torch::Tensor input,
    torch::Tensor output)
{
    int rows = input.size(0);
    int cols = input.size(1);
    
    float* input_ptr = input.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    
    // Use 1024 threads per block for maximum occupancy
    int threads_per_block = 1024;
    if (cols < 1024) {
        threads_per_block = 1;
        while (threads_per_block < cols) {
            threads_per_block *= 2;
        }
    }
    
    int blocks = rows;
    // Shared memory for data + warp sums
    size_t shared_mem_size = threads_per_block * sizeof(float) + 
                            ((threads_per_block + WARP_SIZE - 1) / WARP_SIZE) * sizeof(float);
    
    inclusive_scan_kernel<<<blocks, threads_per_block, shared_mem_size>>>(
        input_ptr, output_ptr, rows, cols);
    
    cudaDeviceSynchronize();
}
"""

# C++ binding code
cpp_source = r"""
#include <torch/extension.h>

void inclusive_scan_forward(
    torch::Tensor input,
    torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("inclusive_scan", &inclusive_scan_forward, "Inclusive scan (cumsum) operation");
}
"""

# Compile the CUDA extension with optimization flags
cumsum_ext = load_inline(
    name='inclusive_scan_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math', '--maxrregcount=32'],
    with_cuda=True,
    verbose=False
)

def functional_model(x, *, dim):
    """
    Computes cumulative sum along the specified dimension using optimized CUDA kernel.
    
    Args:
        x: Input tensor of shape (batch_size, input_shape)
        dim: Dimension along which to compute cumulative sum
    
    Returns:
        Output tensor with cumulative sum computed along dim
    """
    if dim == 1 and x.dtype == torch.float32:
        # Use optimized CUDA kernel for float32 along dimension 1
        output = torch.zeros_like(x)
        cumsum_ext.inclusive_scan(x, output)
        return output
    else:
        # Fallback to PyTorch's cumsum for other cases
        return torch.cumsum(x, dim=dim)

# Configuration parameters
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
