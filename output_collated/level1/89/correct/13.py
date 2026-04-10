# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_075452/code_3.py
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

__global__ void cumsum_kernel(const float* __restrict__ input, float* __restrict__ output, int total_rows, int cols) {
    // Grid-stride loop: process multiple rows with grid-stride pattern
    for (int row = blockIdx.x * blockDim.y + threadIdx.y; row < total_rows; row += gridDim.x * blockDim.y) {
        const float* row_in = input + (long)row * cols;
        float* row_out = output + (long)row * cols;
        
        float running_sum = 0.0f;
        // Process elements in warp-stride pattern within each row
        for (int col = threadIdx.x; col < cols; col += 32) {
            float val = row_in[col];
            
            // Warp-level inclusive scan using shfl_up
            #pragma unroll
            for (int offset = 1; offset < 32; offset <<= 1) {
                float temp = __shfl_up_sync(0xFFFFFFFF, val, offset);
                if (threadIdx.x >= offset) {
                    val += temp;
                }
            }

            // Add carry from previous segment
            val += running_sum;
            
            // Write back to global memory
            row_out[col] = val;

            // Update carry with the sum of this segment (last element of the warp)
            running_sum = __shfl_sync(0xFFFFFFFF, val, 31);
        }
    }
}

void fused_op_forward(int64_t rows, int64_t cols, torch::Tensor input, torch::Tensor output) {
    // Use 2D thread blocks: 32 threads for warp operations, 8 rows per block
    dim3 threads(32, 8);
    // Calculate number of blocks needed, capped to avoid exceeding grid limits
    int num_blocks = (int)((rows + threads.y - 1) / threads.y);
    if (num_blocks > 65535) num_blocks = 65535; // Cap based on max grid dimension for x-axis
    
    dim3 blocks(num_blocks);
    
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
    m.def("fused_op", &fused_op_forward, "Grid-stride fused cumsum");
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
