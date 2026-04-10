# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_074424/code_9.py
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
# CUDA kernel – batch-process 4 rows per block (32 threads × 4 rows)
# -------------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

constexpr int kWarpSize     = 32;
constexpr int kRowsPerBlock = 4;   // number of rows processed by one block

__global__ void cumsum_kernel(
    const float* __restrict__ input,
    float*       __restrict__ output,
    int rows,
    int cols)
{
    // Determine which row this thread is responsible for.
    // blockIdx.x * kRowsPerBlock + threadIdx.y → unique row index
    int row = blockIdx.x * kRowsPerBlock + threadIdx.y;
    if (row >= rows) return;

    const float* row_in  = input  + row * cols;
    float*       row_out = output + row * cols;

    // Lane index within the warp (0..31)
    int lane_id = threadIdx.x;

    float carry = 0.0f;

    // Process the row in chunks of 32 elements (one warp per chunk)
    for (int col_start = 0; col_start < cols; col_start += kWarpSize) {
        int idx = col_start + lane_id;
        float val = (idx < cols) ? row_in[idx] : 0.0f;

        // Warp-level inclusive scan (Hillis-Steele) using shuffle-up
        #pragma unroll
        for (int offset = 1; offset < kWarpSize; offset <<= 1) {
            float tmp = __shfl_up_sync(0xFFFFFFFF, val, offset);
            if (lane_id >= offset) val += tmp;
        }

        // Add carry from the previous segment
        val += carry;

        // Write result back to global memory
        if (idx < cols) row_out[idx] = val;

        // Propagate the segment sum to the next segment
        carry = __shfl_sync(0xFFFFFFFF, val, 31);
    }
}

// Host function that configures the launch
void fused_op_forward(int64_t rows, int64_t cols,
                      torch::Tensor input, torch::Tensor output)
{
    dim3 threads(kWarpSize, kRowsPerBlock);          // 32 × 4 = 128 threads per block
    dim3 blocks((rows + kRowsPerBlock - 1) / kRowsPerBlock);
    cumsum_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(rows),
        static_cast<int>(cols));
}
"""

# -------------------------------------------------------------------------
# C++ binding (pybind11) – exposes the host function to Python
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(int64_t rows, int64_t cols,
                      torch::Tensor input, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward,
          "Batch-processed warp-shuffle cumsum over the last dimension");
}
"""

# -------------------------------------------------------------------------
# Build the inline CUDA extension
# -------------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# -------------------------------------------------------------------------
# Functional wrapper – same interface as the original code
# -------------------------------------------------------------------------
def functional_model(x, *, dim):
    """
    Computes the cumulative sum along the specified dimension.
    The dimension is moved to the last axis, the CUDA kernel is invoked,
    and the result is permuted back if needed.
    """
    original_dtype = x.dtype

    # Ensure we work with FP32 inside the kernel
    x = x.to(torch.float32)

    # If the target dimension is not the last one, transpose it to the last position
    if dim != -1 and dim != x.dim() - 1:
        perm = list(range(x.dim()))
        perm[dim], perm[-1] = perm[-1], perm[dim]
        x = x.permute(*perm)

    x = x.contiguous()
    output = torch.empty_like(x)

    # Number of "rows" = total elements divided by the number of columns (the last dim)
    rows = x.numel() // x.shape[-1]
    cols = x.shape[-1]

    # Launch the batched kernel
    fused_ext.fused_op(rows, cols, x, output)

    # Restore the original dimension order if we had permuted earlier
    if dim != -1 and dim != x.dim() - 1:
        output = output.permute(*perm)

    # Convert back to the original dtype
    return output.to(original_dtype)
