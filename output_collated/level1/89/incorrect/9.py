# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_074424/code_12.py
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

// Each block processes one row. If row is very long, threads loop.
// We use warp-level primitives to minimize shared memory synchronization if possible,
// but for large inputs, registers and coalesced global access are key.
__global__ void fused_op_forward_kernel(const float* __restrict__ input, float* __restrict__ output, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* row_in = input + row * cols;
    float* row_out = output + row * cols;

    float carry = 0.0f;

    // Process row in chunks of 512 total threads (16 warps) to maximize memory throughput
    for (int col_start = 0; col_start < cols; col_start += 512) {
        int idx = col_start + threadIdx.x;
        
        // Load coalesced
        float val = (idx < cols) ? row_in[idx] : 0.0f;

        // Warp-level inclusive scan
        // Threads 0-31 form a warp
        int lane_id = threadIdx.x % 32;
        
        #pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1) {
            float tmp = __shfl_up_sync(0xFFFFFFFF, val, offset);
            if (lane_id >= offset) val += tmp;
        }

        // Add carry from previous warp (or chunk)
        // The last thread in the warp has the sum of the warp
        // Synchronize across warps using __shfl_sync
        
        // This logic handles the cascading carry
        float warp_sum = __shfl_sync(0xFFFFFFFF, val, 31);
        
        // Synchronize carry across the whole block
        // In this implementation, we use a shared memory slot for block-level reduction
        __shared__ float warp_sums[16]; 
        int warp_id = threadIdx.x / 32;
        if (lane_id == 31) warp_sums[warp_id] = val;
        __syncthreads();
        
        if (warp_id > 0) {
            float block_carry = 0.0f;
            for(int i = 0; i < warp_id; ++i) block_carry += warp_sums[i];
            val += block_carry;
        }
        
        val += carry;
        
        if (idx < cols) row_out[idx] = val;
        
        // Prepare carry for next chunk: the sum of the very last element of this chunk
        __syncthreads();
        if (threadIdx.x == 511) carry = val;
        carry = __shfl_sync(0xFFFFFFFF, carry, 31); // Broadcast result to all
    }
}

void fused_op_forward(int64_t rows, int64_t cols, torch::Tensor input, torch::Tensor output) {
    dim3 threads(512);
    dim3 blocks(rows);
    fused_op_forward_kernel<<<blocks, threads>>>(
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
    m.def("fused_op", &fused_op_forward, "High performance fused cumsum");
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
    
    # Handle arbitrary dimension
    permute = False
    if dim != -1 and dim != x.dim() - 1:
        dims = list(range(x.dim()))
        dims[dim], dims[-1] = dims[-1], dims[dim]
        x = x.permute(*dims).contiguous()
        permute = True
    else:
        x = x.contiguous()
    
    rows = x.numel() // x.shape[-1]
    cols = x.shape[-1]
    output = torch.empty_like(x)
    
    fused_ext.fused_op(rows, cols, x, output)
    
    if permute:
        output = output.permute(*dims)
        
    return output.to(original_dtype)
