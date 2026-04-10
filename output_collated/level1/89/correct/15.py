# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_075452/code_10.py
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

# ----------------------------------------------------------------------
# CUDA kernel – warp-level inclusive scan
# ----------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// Warp-level inclusive scan (Hillis-Steele inside a warp)
__device__ __forceinline__ float warp_inclusive_scan(float val, int lane) {
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        float other = __shfl_up_sync(0xffffffff, val, offset);
        if (lane >= offset) val += other;
    }
    return val;
}

__global__ void cumsum_kernel(const float* __restrict__ input,
                              float* __restrict__ output,
                              const int rows,
                              const int cols) {
    const int row = blockIdx.x;
    if (row >= rows) return;

    const float* row_in = input + row * cols;
    float*       row_out = output + row * cols;

    // 256 threads → 8 warps
    const int threads = blockDim.x;
    const int warps   = threads >> 5;               // threads / 32

    // Dynamic shared memory: tile data (threads) + warp sums (warps)
    extern __shared__ float sdata[];
    float* warpSums = sdata + threads;

    // Process the row in tiles of size = threads
    for (int col_start = 0; col_start < cols; col_start += threads) {
        const int tid = threadIdx.x;
        const int idx = col_start + tid;

        // ----- Coalesced load -----
        float val = (idx < cols) ? row_in[idx] : 0.0f;
        sdata[tid] = val;
        __syncthreads();

        // ----- Warp-level inclusive scan -----
        int lane   = tid & 31;
        int warpId = tid >> 5;

        val = warp_inclusive_scan(val, lane);

        // ----- Store warp sums (total of each warp) -----
        if (lane == 31) {  // Last thread in warp writes the total
            warpSums[warpId] = val;
        }
        __syncthreads();

        // ----- Prefix across warps (simple loop, max 7 iterations) -----
        if (warpId > 0) {
            float prefix = 0.0f;
            for (int i = 0; i < warpId; ++i) prefix += warpSums[i];
            val += prefix;
        }

        // ----- Carry from previous tile -----
        if (col_start > 0) {
            float carry = row_out[col_start - 1];
            val += carry;
        }

        // ----- Coalesced write -----
        if (idx < cols) row_out[idx] = val;

        // Ensure the write is visible to the next tile's carry read
        __syncthreads();
    }
}

// Host wrapper
void fused_op_forward(int64_t rows, int64_t cols,
                      torch::Tensor input, torch::Tensor output) {
    const int threads = 256;
    const int warps   = threads >> 5;
    const size_t shmem = (threads + warps) * sizeof(float);
    const int blocks = static_cast<int>(rows);

    cumsum_kernel<<<blocks, threads, shmem>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(rows),
        static_cast<int>(cols)
    );
}
"""

# ----------------------------------------------------------------------
# C++ binding (pybind11)
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(int64_t rows, int64_t cols,
                      torch::Tensor input, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward,
          "Cumulative sum using a warp-level scan");
}
"""

# ----------------------------------------------------------------------
# Build the inline extension
# ----------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# ----------------------------------------------------------------------
# Functional model – entry point for evaluation
# ----------------------------------------------------------------------
def functional_model(x, *, dim):
    # Keep the original dtype and ensure a contiguous float32 tensor
    original_dtype = x.dtype
    x = x.to(torch.float32)

    # If the scan dimension is not the last one, permute it to the end
    if dim != -1 and dim != x.dim() - 1:
        perm = list(range(x.dim()))
        perm[dim], perm[-1] = perm[-1], perm[dim]
        x = x.permute(*perm).contiguous()
    else:
        x = x.contiguous()

    # Allocate output
    output = torch.empty_like(x)

    # Flatten all leading dimensions, keep only the scan dimension
    original_shape = x.shape
    rows = x.numel() // x.shape[-1]
    cols = x.shape[-1]

    # Run the custom CUDA kernel
    fused_ext.fused_op(rows, cols, x, output)

    # Restore the original dimension order if we permuted
    if dim != -1 and dim != len(original_shape) - 1:
        output = output.view(original_shape)
        output = output.permute(*perm)

    return output.to(original_dtype)
