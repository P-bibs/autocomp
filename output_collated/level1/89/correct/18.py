# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_075452/code_18.py
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

__global__ void cumsum_kernel(const float* __restrict__ input, float* __restrict__ output, int rows, int cols) {
    // Each warp processes one row. Block size is 256 (8 warps).
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int row = blockIdx.x * 8 + warp_id;
    
    if (row >= rows) return;

    const float* row_in = input + row * cols;
    float* row_out = output + row * cols;

    float carry = 0.0f;
    
    for (int col_start = 0; col_start < cols; col_start += 32) {
        int idx = col_start + lane_id;
        
        // Coalesced load
        float val = (idx < cols) ? row_in[idx] : 0.0f;

        // Warp-level inclusive scan
        #pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1) {
            float temp = __shfl_up_sync(0xFFFFFFFF, val, offset);
            if (lane_id >= offset) {
                val += temp;
            }
        }

        val += carry;
        
        if (idx < cols) {
            row_out[idx] = val;
        }

        // Broadcast the last value of the warp to all threads in the warp
        carry = __shfl_sync(0xFFFFFFFF, val, 31);
    }
}

void fused_op_forward(int64_t rows, int64_t cols, torch::Tensor input, torch::Tensor output) {
    const int threads_per_block = 256;
    const int warps_per_block = 8;
    // Calculate grid size: ceil(rows / 8)
    int blocks = (rows + warps_per_block - 1) / warps_per_block;
    
    cumsum_kernel<<<blocks, threads_per_block>>>(
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
    m.def("fused_op", &fused_op_forward, "Optimized warp-shfl cumsum");
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
    """
    Computes cumulative sum along a dimension using custom CUDA kernels
    designed for high occupancy on RTX 2080Ti.
    """
    original_dtype = x.dtype
    x = x.to(torch.float32)
    
    # Handle dimension transposition
    original_shape = x.shape
    if dim != -1 and dim != x.dim() - 1:
        permute_dims = list(range(x.dim()))
        permute_dims[dim], permute_dims[-1] = permute_dims[-1], permute_dims[dim]
        x = x.permute(*permute_dims)
    
    x_contig = x.contiguous()
    output = torch.empty_like(x_contig)
    
    # Calculate flattened dimensions
    rows = x_contig.numel() // x_contig.shape[-1]
    cols = x_contig.shape[-1]
    
    # Launch optimized kernel
    fused_ext.fused_op(rows, cols, x_contig, output)
    
    # Restore original shape
    if dim != -1 and dim != x.dim() - 1:
        output = output.permute(*permute_dims)
    
    return output.to(original_dtype).view(original_shape)
