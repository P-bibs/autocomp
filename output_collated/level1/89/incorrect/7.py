# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_074424/code_0.py
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

#define WARP_SIZE 32
#define BLOCK_SIZE 256

__global__ void fused_op_forward_kernel(const float* __restrict__ input, float* __restrict__ output, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    extern __shared__ float shared_data[];

    const float* row_in = input + row * cols;
    float* row_out = output + row * cols;

    float carry = 0.0f;
    
    // Process row in chunks that fit in shared memory
    for (int col_start = 0; col_start < cols; col_start += BLOCK_SIZE) {
        int elements_in_chunk = min(BLOCK_SIZE, cols - col_start);
        
        // Cooperative loading of data into shared memory
        for (int i = threadIdx.x; i < elements_in_chunk; i += BLOCK_SIZE) {
            int idx = col_start + i;
            shared_data[i] = (idx < cols) ? row_in[idx] : 0.0f;
        }
        __syncthreads();

        // Perform warp-level scans on the chunk
        for (int warp_start = 0; warp_start < elements_in_chunk; warp_start += WARP_SIZE) {
            int lane_id = threadIdx.x & 31;
            int idx_in_warp = warp_start + lane_id;
            
            float val = 0.0f;
            if (idx_in_warp < elements_in_chunk) {
                val = shared_data[idx_in_warp];
            }

            // Warp-level inclusive scan using shfl_up
            #pragma unroll
            for (int offset = 1; offset < 32; offset <<= 1) {
                float temp = __shfl_up_sync(0xFFFFFFFF, val, offset);
                if (lane_id >= offset) {
                    val += temp;
                }
            }

            // Add carry from previous segment
            val += carry;
            
            if (idx_in_warp < elements_in_chunk) {
                shared_data[idx_in_warp] = val;
            }
            
            // Update carry with the sum of this warp (last element of the warp)
            if (idx_in_warp < elements_in_chunk) {
                carry = __shfl_sync(0xFFFFFFFF, val, 31);
            }
        }
        __syncthreads();

        // Write results back to global memory
        for (int i = threadIdx.x; i < elements_in_chunk; i += BLOCK_SIZE) {
            int global_idx = col_start + i;
            if (global_idx < cols) {
                row_out[global_idx] = shared_data[i];
            }
        }
        __syncthreads();
    }
}

void fused_op_forward(int64_t rows, int64_t cols, torch::Tensor input, torch::Tensor output) {
    dim3 threads(BLOCK_SIZE);
    dim3 blocks(rows);
    size_t shared_mem_size = BLOCK_SIZE * sizeof(float);
    fused_op_forward_kernel<<<blocks, threads, shared_mem_size>>>(
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
    m.def("fused_op", &fused_op_forward, "Optimized fused cumsum with shared memory");
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
