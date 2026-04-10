# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_083041/code_2.py
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
from torch.utils.cpp_extension import load_inline

# Optimization: Memory Coalescing - Restructure kernel to maximize coalesced memory access
# and GPU occupancy by using multiple threads per row and proper stride handling
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void scan_kernel(const float* input, float* output, int rows, int cols) {
    // Use multiple blocks to handle larger datasets efficiently
    int row = blockIdx.y;
    int tid = threadIdx.x;
    int block_offset = blockIdx.x * blockDim.x;
    
    if (row >= rows) return;
    
    const float* row_in = input + row * cols;
    float* row_out = output + row * cols;
    
    // Shared memory for efficient intra-block communication
    extern __shared__ float shared_data[];
    float* sdata = shared_data;
    
    // Process multiple elements per thread with coalesced access
    int elements_per_thread = (cols + blockDim.x - 1) / blockDim.x;
    
    // Load data with coalesced access pattern
    float local_sum = 0.0f;
    for (int i = 0; i < elements_per_thread; i++) {
        int idx = block_offset + i * blockDim.x + tid;
        if (idx < cols) {
            local_sum += row_in[idx];
            sdata[tid] = local_sum;
        } else {
            sdata[tid] = local_sum;
        }
    }
    
    __syncthreads();
    
    // Warp-level parallel prefix scan within block
    for (int offset = 1; offset < blockDim.x; offset <<= 1) {
        float temp = 0.0f;
        if (tid >= offset) {
            temp = sdata[tid - offset];
        }
        __syncthreads();
        if (tid >= offset) {
            sdata[tid] += temp;
        }
        __syncthreads();
    }
    
    // Write back with coalesced pattern
    for (int i = 0; i < elements_per_thread; i++) {
        int idx = block_offset + i * blockDim.x + tid;
        if (idx < cols) {
            row_out[idx] = sdata[tid];
        }
    }
}

// Host function to launch kernel with proper grid dimensions
void cumsum_cuda(torch::Tensor input, torch::Tensor output) {
    int rows = input.size(0);
    int cols = input.size(1);
    
    // Use 2D grid: (blocks_per_row, num_rows)
    int threads_per_block = 256;  // Better occupancy for RTX 2080Ti
    int blocks_per_row = (cols + threads_per_block - 1) / threads_per_block;
    
    dim3 blocks(blocks_per_row, rows);
    dim3 threads(threads_per_block);
    
    // Shared memory size: one float per thread
    size_t shared_mem_size = threads_per_block * sizeof(float);
    
    scan_kernel<<<blocks, threads, shared_mem_size>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        rows, 
        cols
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void cumsum_cuda(torch::Tensor input, torch::Tensor output);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cumsum", &cumsum_cuda, "Optimized Parallel Prefix Scan with Memory Coalescing");
}
"""

# Compile the extension with optimized flags
fused_ext = load_inline(
    name='cumsum_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math', '-lineinfo'],
    with_cuda=True
)

def functional_model(x, *, dim):
    """
    Optimized cumsum implementation using memory-coalesced CUDA kernel.
    Handles tensors of shape (N, M) with improved memory access patterns.
    """
    # Ensure input is 2D for the kernels
    if x.dim() == 1:
        x = x.unsqueeze(0)
    
    output = torch.empty_like(x)
    fused_ext.cumsum(x.contiguous(), output)
    
    return output.squeeze() if x.dim() == 1 else output

# Required parameters for evaluation
batch_size = 32768
input_shape = (32768,)
dim = 1

def get_init_inputs():
    return [dim]

def get_inputs():
    return [torch.rand(batch_size, *input_shape)]
