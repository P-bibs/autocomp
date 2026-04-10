# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_074424/code_15.py
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

# The CUDA kernel uses a block-wide scan strategy.
# threads: 1024 (standard for modern GPUs).
# Memory: Shared memory to hold the chunk and carry values.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 1024

__global__ void cumsum_kernel(const float* __restrict__ input, float* __restrict__ output, int rows, int cols) {
    __shared__ float sdata[TILE_SIZE];
    
    // Each block processes a range of rows
    for (int row = blockIdx.x; row < rows; row += gridDim.x) {
        float carry = 0.0f;
        int row_offset = row * cols;
        
        for (int col_start = 0; col_start < cols; col_start += TILE_SIZE) {
            int tid = threadIdx.x;
            int idx = col_start + tid;
            
            // 1. Load data into shared memory
            sdata[tid] = (idx < cols) ? input[row_offset + idx] : 0.0f;
            __syncthreads();

            // 2. Intra-warp inclusive scan
            float val = sdata[tid];
            #pragma unroll
            for (int offset = 1; offset < 32; offset <<= 1) {
                float tmp = __shfl_up_sync(0xFFFFFFFF, val, offset);
                if ((tid & 31) >= offset) val += tmp;
            }

            // 3. Store warp results in shared memory for inter-warp scan
            if ((tid & 31) == 31) sdata[tid >> 5] = val;
            __syncthreads();

            // 4. Scan warp sums (only first warp does this)
            if (tid < 32) {
                float warp_sum = (tid < (TILE_SIZE >> 5)) ? sdata[tid] : 0.0f;
                #pragma unroll
                for (int offset = 1; offset < 32; offset <<= 1) {
                    float tmp = __shfl_up_sync(0xFFFFFFFF, warp_sum, offset);
                    if ((tid & 31) >= offset) warp_sum += tmp;
                }
                if (tid < (TILE_SIZE >> 5)) sdata[tid] = warp_sum;
            }
            __syncthreads();

            // 5. Add carry from previous warps and global previous chunk
            if ((tid >> 5) > 0) {
                val += sdata[(tid >> 5) - 1];
            }
            val += carry;

            // 6. Write to output and update global carry
            if (idx < cols) output[row_offset + idx] = val;
            
            // Carry is the total sum of this chunk
            carry = sdata[(TILE_SIZE >> 5) - 1];
            __syncthreads();
        }
    }
}

void fused_op_forward(int64_t rows, int64_t cols, torch::Tensor input, torch::Tensor output) {
    const int threads = TILE_SIZE;
    const int blocks = std::min((int)rows, 65535);
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
    m.def("fused_op", &fused_op_forward, "Fused Block-Shared Scan");
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
    
    # Handle arbitrary dimensions by permuting target dim to last
    permute_dims = None
    if dim != -1 and dim != x.dim() - 1:
        permute_dims = list(range(x.dim()))
        permute_dims[dim], permute_dims[-1] = permute_dims[-1], permute_dims[dim]
        x = x.permute(*permute_dims)
    
    x = x.contiguous()
    output = torch.empty_like(x)
    
    rows = x.numel() // x.shape[-1]
    cols = x.shape[-1]
    
    fused_ext.fused_op(rows, cols, x, output)
    
    if permute_dims is not None:
        output = output.permute(*permute_dims)
    
    return output.to(original_dtype)
