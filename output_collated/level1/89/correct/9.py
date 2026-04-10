# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_073425/code_10.py
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

# The optimized CUDA kernel uses warp-level primitives (__shfl_up_sync) 
# to perform the prefix sum. Register-level communication is significantly 
# faster than shared memory and eliminates the need for barrier synchronizations 
# within the warp, leading to much higher occupancy and throughput.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void cumsum_kernel(const float* __restrict__ input, float* __restrict__ output, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* row_in = input + row * cols;
    float* row_out = output + row * cols;

    float carry = 0.0f;
    // Process the row in chunks of 32 elements (one warp per row)
    for (int col_start = 0; col_start < cols; col_start += 32) {
        int tid = threadIdx.x;
        int idx = col_start + tid;
        
        float val = 0.0f;
        if (idx < cols) {
            val = row_in[idx];
        }

        // Warp-level inclusive prefix sum
        #pragma unroll
        for (int i = 1; i <= 16; i *= 2) {
            float n = __shfl_up_sync(0xffffffff, val, i);
            if (tid >= i) val += n;
        }

        // Add the carry from previous warp iterations
        float final_val = val + carry;
        
        if (idx < cols) {
            row_out[idx] = final_val;
        }
        
        // Broadcast the last element of the current warp to update the carry variable
        // for the next iteration (next 32 elements). Note: Thread 31 holds the sum of the warp.
        carry += __shfl_sync(0xffffffff, val, 31);
    }
}

void fused_op_forward(int64_t rows, int64_t cols, torch::Tensor input, torch::Tensor output) {
    // We launch one warp per row.
    dim3 threads(32);
    dim3 blocks(rows);
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
    m.def("fused_op", &fused_op_forward, "Fused warp-level optimized cumsum");
}
"""

# Compile the extension with high-performance flags
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, dim):
    # Standardize input for the kernel: Float32 and contiguous
    original_dtype = x.dtype
    x = x.to(torch.float32)
    
    # Target dimension handling: if not trailing, permute to put dim at the end
    original_shape = x.shape
    if dim != -1 and dim != x.dim() - 1:
        dims = list(range(x.dim()))
        dims[dim], dims[-1] = dims[-1], dims[dim]
        x = x.permute(dims).contiguous()
    else:
        x = x.contiguous()
    
    rows = x.numel() // x.shape[-1]
    cols = x.shape[-1]
    
    output = torch.empty_like(x)
    
    # Launch kernel
    fused_ext.fused_op(rows, cols, x, output)
    
    # Restore original shape/permute if necessary
    if dim != -1 and dim != len(original_shape) - 1:
        output = output.view([original_shape[i] for i in dims])
        # Reverse the permutation effectively
        inv_dims = [0] * len(dims)
        for i, d in enumerate(dims):
            inv_dims[d] = i
        output = output.permute(inv_dims)
        
    return output.to(original_dtype)
