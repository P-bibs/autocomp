# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_075452/code_9.py
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
    int row = blockIdx.x;
    if (row >= rows) return;

    extern __shared__ float s_data[];
    float* s_warp_sums = s_data + blockDim.x;  // Additional space for warp sums
    
    const float* row_in = input + row * cols;
    float* row_out = output + row * cols;

    const int warp_size = 32;
    const int num_warps = blockDim.x / warp_size;
    const int lane_id = threadIdx.x & 31;
    const int warp_id = threadIdx.x / 32;
    
    float carry = 0.0f;
    
    // Process row in chunks of blockDim.x elements
    for (int col_start = 0; col_start < cols; col_start += blockDim.x) {
        // Load data into shared memory - coalesced access
        int gidx = col_start + threadIdx.x;
        s_data[threadIdx.x] = (gidx < cols) ? row_in[gidx] : 0.0f;
        __syncthreads();
        
        // Each warp performs warp-level scan on its 32-element chunk
        float val = s_data[threadIdx.x];
        
        // Warp-level inclusive scan using shfl_up
        #pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1) {
            float temp = __shfl_up_sync(0xFFFFFFFF, val, offset);
            if (lane_id >= offset) {
                val += temp;
            }
        }
        
        // Store warp results back to shared memory
        s_data[threadIdx.x] = val;
        
        // Store the sum of each warp (last element of each warp)
        if (lane_id == 31) {
            s_warp_sums[warp_id] = val;
        }
        __syncthreads();
        
        // Scan the warp sums to propagate carries between warps
        if (warp_id == 0) {
            float warp_val = s_warp_sums[lane_id < num_warps ? lane_id : 0];
            
            // Warp-level scan on warp sums
            #pragma unroll
            for (int offset = 1; offset < 32; offset <<= 1) {
                float temp = __shfl_up_sync(0xFFFFFFFF, warp_val, offset);
                if (lane_id >= offset && lane_id < num_warps) {
                    warp_val += temp;
                }
            }
            
            if (lane_id < num_warps) {
                s_warp_sums[lane_id] = warp_val;
            }
        }
        __syncthreads();
        
        // Add carry from previous segment and inter-warp carries
        if (warp_id > 0) {
            val += s_warp_sums[warp_id - 1];
        }
        val += carry;
        
        // Store final result
        s_data[threadIdx.x] = val;
        __syncthreads();
        
        // Write results to global memory
        if (gidx < cols) {
            row_out[gidx] = s_data[threadIdx.x];
        }
        
        // Update carry for next iteration
        if (threadIdx.x == blockDim.x - 1) {
            carry = s_data[threadIdx.x];
        }
        __syncthreads();
    }
}

void fused_op_forward(int64_t rows, int64_t cols, torch::Tensor input, torch::Tensor output) {
    // Use 256 threads per block (8 warps)
    const int block_size = 256;
    // Shared memory: block_size floats for data + 8 floats for warp sums
    int shared_mem_size = (block_size + block_size/32) * sizeof(float);
    
    dim3 threads(block_size);
    dim3 blocks(rows);
    
    cumsum_kernel<<<blocks, threads, shared_mem_size>>>(
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
    m.def("fused_op", &fused_op_forward, "Optimized fused warp-shfl cumsum");
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
