# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_074424/code_4.py
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

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

// Warp-level inclusive scan using shfl_up_sync
__device__ __forceinline__ float warp_inclusive_scan(float val) {
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        float temp = __shfl_up_sync(0xffffffff, val, offset);
        if ((threadIdx.x & 31) >= offset) {
            val += temp;
        }
    }
    return val;
}

// Warp-level exclusive scan derived from inclusive scan
__device__ __forceinline__ float warp_exclusive_scan(float val) {
    float res = warp_inclusive_scan(val);
    res = __shfl_up_sync(0xffffffff, res, 1);
    return (threadIdx.x & 31) ? res : 0.0f;
}

__global__ void cumsum_kernel(const float* __restrict__ input, float* __restrict__ output, int rows, int cols) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    
    if (bid >= rows) return;

    const float* row_in = input + bid * cols;
    float* row_out = output + bid * cols;

    // Shared memory to hold warp sums for inter-warp scan
    __shared__ float warp_sums[32];  // max 32 warps per block assumed
    __shared__ float block_sum;
    
    float carry = 0.0f;

    // Grid-stride loop over columns
    for (int col_start = 0; col_start < cols; col_start += blockDim.x) {
        int idx = col_start + tid;
        float val = (idx < cols) ? row_in[idx] : 0.0f;
        
        // Step 1: Warp-level exclusive scan
        float scan_result = warp_exclusive_scan(val);
        
        // Step 2: Get the total sum of this warp (last thread's inclusive scan result)
        float warp_sum = __shfl_sync(0xffffffff, warp_inclusive_scan(val), 31);
        
        // Step 3: Store warp sum in shared memory
        if (lane_id == 31) {
            warp_sums[warp_id] = warp_sum;
        }
        __syncthreads();
        
        // Step 4: First warp scans the warp sums to get inter-warp carries
        if (warp_id == 0) {
            float temp = warp_sums[lane_id < (blockDim.x / 32) ? lane_id : 0];
            temp = (lane_id < (blockDim.x / 32)) ? temp : 0.0f;
            float warp_carry = warp_exclusive_scan(temp);
            warp_sums[lane_id] = warp_carry;
        }
        __syncthreads();
        
        // Step 5: Add inter-warp carry to local warp result
        float inter_warp_carry = (warp_id > 0) ? warp_sums[warp_id] : 0.0f;
        scan_result += inter_warp_carry + carry;
        
        // Step 6: Write result to global memory
        if (idx < cols) {
            row_out[idx] = scan_result + val;  // Inclusive scan
        }
        
        // Step 7: Update carry for next iteration
        __syncthreads();
        if (tid == blockDim.x - 1) {
            block_sum = row_out[idx < cols ? idx : 0];
        }
        __syncthreads();
        carry = block_sum;
    }
}

void fused_op_forward(int64_t rows, int64_t cols, torch::Tensor input, torch::Tensor output) {
    const int threads = 256;
    const int blocks = rows;
    
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
    m.def("fused_op", &fused_op_forward, "Optimized cumsum implementation");
}
"""

fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, dim=-1):
    original_dtype = x.dtype
    x = x.to(torch.float32)
    
    if dim != -1 and dim != x.dim() - 1:
        # Simplification: Only support last dim for optimized kernel
        raise NotImplementedError("Only dim=-1 supported for this optimized implementation")
    
    x = x.contiguous()
    rows = x.numel() // x.shape[-1]
    cols = x.shape[-1]
    output = torch.empty_like(x)
    
    fused_ext.fused_op(rows, cols, x, output)
    return output.to(original_dtype)
