# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_081907/code_21.py
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

# -------------------------------------------------------------------------
# CUDA kernel – Optimized for NVIDIA Turing (RTX 2080Ti)
# Uses 8 warps per block to improve SM occupancy. 
# Each warp handles one row independently, maximizing coalesced memory access.
# -------------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float warp_inclusive_scan(float val) {
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        float tmp = __shfl_up_sync(0xFFFFFFFF, val, offset);
        if ((threadIdx.x & 31) >= offset) {
            val += tmp;
        }
    }
    return val;
}

__global__ void cumsum_kernel(const float* __restrict__ input,
                              float* __restrict__ output,
                              const int rows,
                              const int cols) {
    // Each warp (threadIdx.y) works on one row.
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= rows) return;

    const float* row_in  = input + row * cols;
    float*       row_out = output + row * cols;

    float carry = 0.0f;
    // Process columns in blocks of 32 (coalesced 128-byte alignment per warp)
    for (int col_start = 0; col_start < cols; col_start += 32) {
        int idx = col_start + (threadIdx.x & 31);
        float val = (idx < cols) ? row_in[idx] : 0.0f;

        val = warp_inclusive_scan(val);
        val += carry;

        if (idx < cols) {
            row_out[idx] = val;
        }

        // Sync carry across the warp: get result of the last thread
        carry = __shfl_sync(0xFFFFFFFF, val, 31);
    }
}

void fused_op_forward(int64_t rows, int64_t cols, torch::Tensor input, torch::Tensor output) {
    // 8 warps per block (256 threads) provides good balance between occupancy and shared memory
    const int warps_per_block = 8;
    const int blocks = (rows + warps_per_block - 1) / warps_per_block;
    
    dim3 block(32, warps_per_block);
    dim3 grid(blocks);

    cumsum_kernel<<<grid, block>>>(
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

# Compile the extension
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, dim):
    """
    Computes cumulative sum along a dimension using a high-occupancy CUDA kernel.
    """
    original_dtype = x.dtype
    x = x.to(torch.float32)
    
    # Re-order dimensions if necessary to ensure target dim is trailing
    if dim != -1 and dim != x.dim() - 1:
        new_order = list(range(x.dim()))
        new_order[dim], new_order[-1] = new_order[-1], new_order[dim]
        x = x.permute(*new_order).contiguous()
    else:
        x = x.contiguous()
    
    # Get flattened rows and columns count
    cols = x.shape[-1]
    rows = x.numel() // cols
    
    output = torch.empty_like(x)
    
    # Execute optimized CUDA kernel
    fused_ext.fused_op(rows, cols, x, output)
    
    # Restore original dimension ordering if permuted
    if dim != -1 and dim != x.dim() - 1:
        # Inverse permutation
        output = output.permute(*new_order)
        
    return output.to(original_dtype)
