# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_081907/code_9.py
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
# CUDA kernel – now runs 8 warps per block (256 threads) for higher occupancy
# -------------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// Warp-level inclusive scan using shuffle instructions
__device__ __forceinline__ float warp_inclusive_scan(float val,
                                                      const unsigned int mask = 0xFFFFFFFF) {
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        float tmp = __shfl_up_sync(mask, val, offset);
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
    // blockDim = (32, 8) → 8 warps per block
    const int warpId   = threadIdx.y;                // 0 … 7
    const int lane     = threadIdx.x & 31;           // 0 … 31
    const int row      = blockIdx.x * blockDim.y + warpId;

    if (row >= rows) return;

    const float* row_in  = input + row * cols;
    float*       row_out = output + row * cols;

    float carry = 0.0f;
    // Process the row in chunks of 32 elements (one warp per chunk)
    for (int col_start = 0; col_start < cols; col_start += 32) {
        int idx = col_start + lane;
        float val = (idx < cols) ? row_in[idx] : 0.0f;

        // inclusive scan inside the warp
        val = warp_inclusive_scan(val);

        // add carry from previous segment
        val += carry;

        // write back to global memory
        if (idx < cols) {
            row_out[idx] = val;
        }

        // carry for the next segment = last element of this warp
        carry = __shfl_sync(0xFFFFFFFF, val, 31);
    }
}

// Host routine – chooses a block size of 256 threads (8 warps)
void fused_op_forward(int64_t rows, int64_t cols,
                      torch::Tensor input, torch::Tensor output) {
    const int threadsPerBlock = 256;                // 32 threads * 8 warps
    const int warpsPerBlock   = threadsPerBlock / 32;
    const int blocks = (rows + warpsPerBlock - 1) / warpsPerBlock; // ceil(rows/8)

    dim3 blockDim(32, warpsPerBlock);
    dim3 gridDim(blocks);

    cumsum_kernel<<<gridDim, blockDim>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(rows),
        static_cast<int>(cols)
    );
}
"""

# -------------------------------------------------------------------------
# C++ binding (PYBIND11)
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(int64_t rows, int64_t cols,
                      torch::Tensor input, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward,
          "Fused warp-shuffle cumsum with higher occupancy");
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
    original_dtype = x.dtype
    x = x.to(torch.float32)

    # Permute so that the target dimension becomes the last one
    if dim != -1 and dim != x.dim() - 1:
        perm = list(range(x.dim()))
        perm[dim], perm[-1] = perm[-1], perm[dim]
        x = x.permute(*perm)

    x = x.contiguous()
    output = torch.empty_like(x)

    rows = x.numel() // x.shape[-1]
    cols = x.shape[-1]

    # Call the high-occupancy CUDA kernel
    fused_ext.fused_op(rows, cols, x, output)

    # Restore original dimension ordering
    if dim != -1 and dim != x.dim() - 1:
        output = output.permute(*perm)

    return output.to(original_dtype)
