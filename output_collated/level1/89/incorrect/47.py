# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_083041/code_21.py
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

# CUDA kernel with warp-level scan and optimized carry propagation
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void cumsum_kernel(const float* __restrict__ input, float* __restrict__ output, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* row_in = input + row * cols;
    float* row_out = output + row * cols;

    float carry = 0.0f;
    for (int col_start = 0; col_start < cols; col_start += blockDim.x) {
        int tid = threadIdx.x;
        int idx = col_start + tid;
        
        float val = (idx < cols) ? row_in[idx] : 0.0f;

        // Perform inclusive scan within each warp
        for (int offset = 1; offset < 32; offset <<= 1) {
            float tmp = __shfl_up_sync(0xFFFFFFFF, val, offset);
            if (tid % 32 >= offset) val += tmp;
        }

        // Add carry that was computed from previous warp
        val += carry;

        if (idx < cols) row_out[idx] = val;

        // Synchronize and update carry for the next tile
        // The last thread in the block calculates the total sum of this tile
        // and broadcasts it to all warps for the next iteration
        float tile_sum = __shfl_sync(0xFFFFFFFF, val, 31);
        __syncthreads();
        
        if (tid == 0) {
            carry += tile_sum;
        }
        carry = __shfl_sync(0xFFFFFFFF, carry, 0); 
        __syncthreads();
    }
}

void fused_op_forward(int64_t rows, int64_t cols, torch::Tensor input, torch::Tensor output) {
    // 256 threads per block is optimal for occupancy on 2080Ti
    const int threads = 256;
    cumsum_kernel<<<rows, threads>>>(
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
    m.def("fused_op", &fused_op_forward, "Fused cumsum execution via warp-level primitives");
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
    
    # Handle dimension permutation
    original_shape = x.shape
    if dim != -1 and dim != x.dim() - 1:
        dims = list(range(x.dim()))
        dims[dim], dims[-1] = dims[-1], dims[dim]
        x = x.permute(*dims).contiguous()
    else:
        x = x.contiguous()
    
    output = torch.empty_like(x)
    rows = x.numel() // x.shape[-1]
    cols = x.shape[-1]
    
    fused_ext.fused_op(rows, cols, x, output)
    
    # Restore shape
    if dim != -1 and dim != len(original_shape) - 1:
        output = output.permute(*dims)
        
    return output.to(original_dtype)
