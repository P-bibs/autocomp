# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_080721/code_15.py
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

// Each block processes rows in a grid-stride loop.
// We use 128 threads per block (4 warps) to improve occupancy and latency hiding.
__global__ void cumsum_kernel(const float* __restrict__ input, float* __restrict__ output, int rows, int cols) {
    int tid = threadIdx.x;
    int lane_id = tid & 31;
    
    // Each block processes a subset of rows
    for (int row = blockIdx.x; row < rows; row += gridDim.x) {
        const float* row_in = input + row * cols;
        float* row_out = output + row * cols;

        float carry = 0.0f;
        // Process row in chunks of 32 (one warp per chunk)
        // If blockDim > 32, we assign row-segments to different warps in the same block
        for (int col_start = 0; col_start < cols; col_start += (blockDim.x / 32) * 32) {
            int warp_id = tid / 32;
            int col_offset = col_start + (warp_id * 32);
            int idx = col_offset + lane_id;
            
            float val = (idx < cols) ? row_in[idx] : 0.0f;

            #pragma unroll
            for (int offset = 1; offset < 32; offset <<= 1) {
                float temp = __shfl_up_sync(0xFFFFFFFF, val, offset);
                if (lane_id >= offset) val += temp;
            }

            val += carry;
            if (idx < cols) row_out[idx] = val;
            
            // Carry is the last element of the current warp's segment
            float warp_sum = __shfl_sync(0xFFFFFFFF, val, 31);
            
            // Broadcast carry to next segments
            carry += warp_sum;
        }
    }
}

void fused_op_forward(int64_t rows, int64_t cols, torch::Tensor input, torch::Tensor output) {
    const int threads = 128;
    const int blocks = std::min((int)rows, 512); 
    cumsum_kernel<<<blocks, threads>>>(
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
    m.def("fused_op", &fused_op_forward, "Optimized fused batch cumsum");
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
    
    # Handle dimension permuting to ensure the scan is always on the last dimension
    if dim != -1 and dim != x.dim() - 1:
        permute_dims = list(range(x.dim()))
        permute_dims[dim], permute_dims[-1] = permute_dims[-1], permute_dims[dim]
        x = x.permute(*permute_dims)
    
    x_contig = x.contiguous()
    output = torch.empty_like(x_contig)
    
    rows = x_contig.numel() // x_contig.shape[-1]
    cols = x_contig.shape[-1]
    
    fused_ext.fused_op(rows, cols, x_contig, output)
    
    if dim != -1 and dim != x.dim() - 1:
        # Undo permutation
        output = output.permute(*permute_dims)
        
    return output.to(original_dtype)
