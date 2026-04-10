# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_083041/code_9.py
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

# Optimized two-pass kernel using warp-level primitives and decoupled carry propagation
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// Warp-level inclusive scan using shfl operations
__device__ __forceinline__ float warp_inclusive_scan(float val) {
    #pragma unroll
    for (int offset = 1; offset < 32; offset *= 2) {
        float other = __shfl_up_sync(0xffffffff, val, offset);
        if ((threadIdx.x & 31) >= offset) {
            val += other;
        }
    }
    return val;
}

// Warp-level exclusive scan
__device__ __forceinline__ float warp_exclusive_scan(float val) {
    float res = warp_inclusive_scan(val);
    return __shfl_up_sync(0xffffffff, res, 1);
}

__global__ void cumsum_pass1_kernel(
    const float* input, float* output, float* block_sums,
    int rows, int cols, int tile_size) {
    
    extern __shared__ float sdata[];
    
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* row_in = input + row * cols;
    float* row_out = output + row * cols;
    
    int num_warps = (blockDim.x + 31) / 32;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    // Process row in tiles
    for (int tile_start = 0; tile_start < cols; tile_start += tile_size) {
        int tid = threadIdx.x;
        int idx = tile_start + tid;
        
        // Coalesced load
        float val = (idx < cols) ? row_in[idx] : 0.0f;
        sdata[tid] = val;
        __syncthreads();

        // Step 1: Warp-level inclusive scan
        float scan_val = warp_inclusive_scan(val);
        
        // Step 2: Store warp sums
        if (lane_id == 31) {
            sdata[num_warps + warp_id] = scan_val;
        }
        __syncthreads();
        
        // Step 3: Scan warp sums (done by first warp)
        if (warp_id == 0) {
            float warp_sum = (lane_id < num_warps) ? sdata[num_warps + lane_id] : 0.0f;
            warp_sum = warp_inclusive_scan(warp_sum);
            if (lane_id < num_warps) {
                sdata[num_warps + lane_id] = warp_sum;
            }
        }
        __syncthreads();
        
        // Step 4: Add propagated warp sums
        if (warp_id > 0) {
            scan_val += sdata[num_warps + warp_id - 1];
        }
        
        // Store result
        if (idx < cols) {
            row_out[idx] = scan_val;
        }
        __syncthreads();
    }
    
    // Store final value of this row for carry propagation
    if (threadIdx.x == blockDim.x - 1) {
        block_sums[row] = row_out[cols - 1];
    }
}

__global__ void cumsum_pass2_kernel(
    float* output, const float* block_sums,
    int rows, int cols, int tile_size) {
    
    int row = blockIdx.x;
    if (row >= rows) return;
    
    float* row_out = output + row * cols;
    
    // Load carry from previous row
    float carry = (row > 0) ? block_sums[row - 1] : 0.0f;
    
    // Add carry to all elements of this row
    for (int tile_start = 0; tile_start < cols; tile_start += tile_size) {
        int idx = tile_start + threadIdx.x;
        if (idx < cols) {
            row_out[idx] += carry;
        }
        __syncthreads(); // Ensure all threads in block complete before next tile
    }
}

void fused_op_forward(int64_t rows, int64_t cols, torch::Tensor input, torch::Tensor output) {
    const int threads = 256;  // Multiple of warp size
    const int blocks = rows;
    const int tile_size = threads;
    const size_t shared_mem = (threads + (threads/32 + 1)) * sizeof(float); // Extra space for warp sums
    
    // Allocate temporary buffer for block sums
    torch::Tensor block_sums = torch::empty({rows}, input.options());
    
    // Pass 1: Local scans with carry computation
    cumsum_pass1_kernel<<<blocks, threads, shared_mem>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(),
        block_sums.data_ptr<float>(),
        (int)rows, 
        (int)cols,
        tile_size
    );
    
    // Pass 2: Carry propagation
    cumsum_pass2_kernel<<<blocks, threads>>>(
        output.data_ptr<float>(),
        block_sums.data_ptr<float>(),
        (int)rows, 
        (int)cols,
        tile_size
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_op_forward(int64_t rows, int64_t cols, torch::Tensor input, torch::Tensor output);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Two-pass optimized cumsum with warp-level primitives");
}
"""

fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math', '-maxrregcount=64'],
    with_cuda=True
)

def functional_model(x, *, dim):
    # Ensure contiguous and float32 for high-performance alignment
    original_dtype = x.dtype
    x = x.to(torch.float32)
    
    # Handle dimension: kernel assumes scan across columns (dim=-1)
    if dim != -1 and dim != x.dim() - 1:
        permute_dims = list(range(x.dim()))
        permute_dims[dim], permute_dims[-1] = permute_dims[-1], permute_dims[dim]
        x = x.permute(*permute_dims)
    
    x = x.contiguous()
    output = torch.empty_like(x)
    
    original_shape = x.shape
    rows = x.numel() // x.shape[-1]
    cols = x.shape[-1]
    
    fused_ext.fused_op(rows, cols, x, output)
    
    # Restore shape/permute if needed
    if dim != -1 and dim != len(original_shape) - 1:
        output = output.view(original_shape)
        inv_permute_dims = [0] * len(permute_dims)
        for i, p in enumerate(permute_dims):
            inv_permute_dims[p] = i
        output = output.permute(*inv_permute_dims)
    
    return output.to(original_dtype)
