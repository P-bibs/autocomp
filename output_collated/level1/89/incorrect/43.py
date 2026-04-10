# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_083041/code_8.py
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

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

#define BLOCK_SIZE 256
#define WARP_SIZE 32

__global__ void cumsum_kernel_optimized(
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows,
    int cols) {
    
    // Use shared memory for block-level scan
    typedef cub::BlockScan<float, BLOCK_SIZE> BlockScan;
    __shared__ typename BlockScan::TempStorage temp_storage;
    
    int row = blockIdx.x;
    if (row >= rows) return;
    
    const float* row_in = input + row * cols;
    float* row_out = output + row * cols;
    
    // Process the row in chunks of BLOCK_SIZE
    float carry = 0.0f;
    for (int col_start = 0; col_start < cols; col_start += BLOCK_SIZE) {
        int elements_in_tile = min(BLOCK_SIZE, cols - col_start);
        int tid = threadIdx.x;
        
        // Load data into shared memory/register
        float thread_data = (tid < elements_in_tile) ? 
                            row_in[col_start + tid] : 0.0f;
        
        // Perform inclusive prefix sum
        float scan_result;
        BlockScan(temp_storage).InclusiveScan(thread_data, scan_result, cub::Sum());
        __syncthreads(); // Sync for reusing temp_storage
        
        // Add carry from previous tile
        scan_result += carry;
        
        // Write back to global memory
        if (tid < elements_in_tile) {
            row_out[col_start + tid] = scan_result;
        }
        
        // Update carry with last element of this tile
        if (tid == elements_in_tile - 1) {
            carry = scan_result;
        }
        __syncthreads(); // Ensure all threads see updated carry
    }
}

void fused_op_forward(int64_t rows, int64_t cols, torch::Tensor input, torch::Tensor output) {
    dim3 threads(BLOCK_SIZE);
    dim3 blocks(rows);
    
    cumsum_kernel_optimized<<<blocks, threads>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        (int)rows, 
        (int)cols
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_op_forward(int64_t rows, int64_t cols, torch::Tensor input, torch::Tensor output);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Optimized block-wide cumsum using CUB");
}
"""

fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, dim):
    original_dtype = x.dtype
    x = x.to(torch.float32)
    
    # Handle dimension: move target dim to last position
    if dim != -1 and dim != x.dim() - 1:
        permute_dims = list(range(x.dim()))
        permute_dims[dim], permute_dims[-1] = permute_dims[-1], permute_dims[dim]
        x = x.permute(*permute_dims)
    
    x = x.contiguous()
    output = torch.empty_like(x)
    
    rows = x.numel() // x.shape[-1]
    cols = x.shape[-1]
    
    fused_ext.fused_op(rows, cols, x, output)
    
    # Restore original dimension order if permuted
    if dim != -1 and dim != x.dim() - 1:
        output = output.permute(*permute_dims)
    
    return output.to(original_dtype)
