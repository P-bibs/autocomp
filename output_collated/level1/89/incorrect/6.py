# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_074424/code_3.py
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

#define TILE_DIM 1024

__global__ void cumsum_kernel(const float* __restrict__ input, float* __restrict__ output, int rows, int cols) {
    __shared__ float sdata[TILE_DIM];
    __shared__ float warp_sums[(TILE_DIM + 31) / 32];
    
    for (int row = blockIdx.x; row < rows; row += gridDim.x) {
        float carry = 0.0f;
        int row_offset = row * cols;
        
        for (int col_start = 0; col_start < cols; col_start += TILE_DIM) {
            int tid = threadIdx.x;
            int idx = col_start + tid;
            
            // Load tile into shared memory
            sdata[tid] = (idx < cols) ? input[row_offset + idx] : 0.0f;
            __syncthreads();

            // Perform warp-level inclusive scan
            float val = sdata[tid];
            for (int offset = 1; offset < 32; offset <<= 1) {
                float temp = __shfl_up_sync(0xFFFFFFFF, val, offset);
                if ((tid & 31) >= offset) {
                    val += temp;
                }
            }
            
            // Store the last value of each warp (partial sums)
            int warp_id = tid / 32;
            if ((tid & 31) == 31) {
                warp_sums[warp_id] = val;
            }
            __syncthreads();

            // Scan the warp partial sums
            if (tid < (TILE_DIM + 31) / 32) {
                float sum = warp_sums[tid];
                for (int offset = 1; offset < 32; offset <<= 1) {
                    float temp = __shfl_up_sync(0xFFFFFFFF, sum, offset);
                    if ((tid & 31) >= offset) {
                        sum += temp;
                    }
                }
                warp_sums[tid] = sum;
            }
            __syncthreads();

            // Add carry from previous warps
            if (warp_id > 0) {
                val += warp_sums[warp_id - 1];
            }
            
            // Add global carry from previous tiles
            val += carry;
            
            // Write result back to global memory
            if (idx < cols) {
                output[row_offset + idx] = val;
            }
            
            // Update carry with the total sum of this tile
            if (tid == TILE_DIM - 1 || (col_start + TILE_DIM) >= cols) {
                carry += (col_start + TILE_DIM < cols) ? 
                         ((col_start + TILE_DIM - 1 < cols) ? val : warp_sums[(cols - col_start - 1) / 32]) : 
                         val;
            }
            __syncthreads();
        }
    }
}

void fused_op_forward(int rows, int cols, torch::Tensor input, torch::Tensor output) {
    if (cols <= 0) return;
    
    dim3 block(1024);
    dim3 grid(min(rows, 65535));
    
    cumsum_kernel<<<grid, block>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        rows, 
        cols
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(int rows, int cols, torch::Tensor input, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Optimized Block-Wide Cooperative Cumsum");
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
    
    if dim != -1 and dim != x.dim() - 1:
        permute_dims = list(range(x.dim()))
        permute_dims[dim], permute_dims[-1] = permute_dims[-1], permute_dims[dim]
        x = x.permute(*permute_dims)
    
    x = x.contiguous()
    output = torch.empty_like(x)
    
    rows = x.numel() // x.shape[-1]
    cols = x.shape[-1]
    
    fused_ext.fused_op(rows, cols, x, output)
    
    if dim != -1 and dim != x.dim() - 1:
        output = output.permute(*permute_dims)
    
    return output.to(original_dtype)
