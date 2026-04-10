# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_075452/code_12.py
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
    // Each block processes one or more rows. 
    // Given the grid-stride design, we distribute row-processing across the grid.
    for (int row = blockIdx.x; row < rows; row += gridDim.x) {
        float carry = 0.0f;
        const float* row_in = input + (size_t)row * cols;
        float* row_out = output + (size_t)row * cols;

        // Process row in chunks of 32 (warp size)
        for (int col_start = 0; col_start < cols; col_start += 32) {
            int lane_id = threadIdx.x;
            int idx = col_start + lane_id;
            
            float val = (idx < cols) ? row_in[idx] : 0.0f;

            // Warp-level inclusive scan
            #pragma unroll
            for (int offset = 1; offset < 32; offset <<= 1) {
                float temp = __shfl_up_sync(0xFFFFFFFF, val, offset);
                if (lane_id >= offset) {
                    val += temp;
                }
            }

            // Carry over sum from previous warp segment
            val += carry;
            
            if (idx < cols) {
                row_out[idx] = val;
            }

            // The last lane of the warp now holds the total sum of this chunk
            carry = __shfl_sync(0xFFFFFFFF, val, 31);
        }
    }
}

void fused_op_forward(int64_t rows, int64_t cols, torch::Tensor input, torch::Tensor output) {
    // 32 threads per block is optimal for the shfl_up approach per row
    int threads = 32;
    // Map each row to a block, capped by device limits (65535 is safe, but we use a reasonable cap)
    int blocks = (int)std::min((int64_t)65535, rows);
    
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
    m.def("fused_op", &fused_op_forward, "Fused warp-shfl cumsum");
}
"""

# Compile the extension JIT
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, dim):
    """
    Computes the cumulative sum along the specified dimension using a custom CUDA kernel.
    """
    original_dtype = x.dtype
    x = x.to(torch.float32)
    
    # Handle dimension permuting to ensure the target dimension is the last one
    if dim != -1 and dim != x.dim() - 1:
        permute_dims = list(range(x.dim()))
        permute_dims[dim], permute_dims[-1] = permute_dims[-1], permute_dims[dim]
        x = x.permute(*permute_dims)
    
    x = x.contiguous()
    output = torch.empty_like(x)
    
    # Flatten the leading dimensions into "rows"
    rows = x.numel() // x.shape[-1]
    cols = x.shape[-1]
    
    # Launch CUDA kernel
    fused_ext.fused_op(rows, cols, x, output)
    
    # Reverse permutation if applied
    if dim != -1 and dim != x.dim() - 1:
        output = output.permute(*permute_dims)
    
    return output.to(original_dtype)
