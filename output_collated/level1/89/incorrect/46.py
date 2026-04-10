# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_083041/code_16.py
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
    // Each warp handles one row. Grid-stride loop moves the warp to the next row.
    // blockDim.x is fixed at 32.
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    int stride = gridDim.x * blockDim.y;

    for (; row < rows; row += stride) {
        const float* row_in = input + row * cols;
        float* row_out = output + row * cols;

        float carry = 0.0f;
        // Process row in chunks of 32 (warp capacity)
        for (int col_start = 0; col_start < cols; col_start += 32) {
            int idx = col_start + threadIdx.x;
            float val = (idx < cols) ? row_in[idx] : 0.0f;

            // Exclusive scan using shuffle: warp-level inclusive scan
            #pragma unroll
            for (int offset = 1; offset < 32; offset <<= 1) {
                float temp = __shfl_up_sync(0xFFFFFFFF, val, offset);
                if (threadIdx.x >= offset) {
                    val += temp;
                }
            }

            val += carry;
            if (idx < cols) {
                row_out[idx] = val;
            }
            
            // Broadcast the last element of this warp to all threads in current warp
            carry = __shfl_sync(0xFFFFFFFF, val, 31);
        }
    }
}

void fused_op_forward(int64_t rows, int64_t cols, torch::Tensor input, torch::Tensor output) {
    // 32 threads per warp. We map rows to warps.
    // Using 4 warps per block (128 threads total) to improve SM occupancy.
    const int threads_per_block = 128;
    const int warps_per_block = 4; // 128 / 32
    
    int grid_size = (rows + warps_per_block - 1) / warps_per_block;
    grid_size = std::min(grid_size, 65535); // Clamp to hardware limits

    cumsum_kernel<<<grid_size, threads_per_block>>>(
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
    m.def("fused_op", &fused_op_forward, "Optimized Warp-Shfl Cumsum");
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
    
    # Handle dimension transposition to ensure target dim is always the last dimension
    if dim != -1 and dim != x.dim() - 1:
        permute_dims = list(range(x.dim()))
        permute_dims[dim], permute_dims[-1] = permute_dims[-1], permute_dims[dim]
        x = x.permute(*permute_dims)
    
    # Ensure contiguous for coalesced memory access
    x_contig = x.contiguous()
    output = torch.empty_like(x_contig)
    
    rows = x_contig.numel() // x_contig.shape[-1]
    cols = x_contig.shape[-1]
    
    fused_ext.fused_op(rows, cols, x_contig, output)
    
    if dim != -1 and dim != x.dim() - 1:
        output = output.permute(*permute_dims)
    
    return output.to(original_dtype)
