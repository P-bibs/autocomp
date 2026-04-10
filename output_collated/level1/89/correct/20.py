# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_075452/code_22.py
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
# CUDA kernel – optimized warp‑level inclusive scan
# ----------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// Warp‑level inclusive scan using __shfl_up_sync
__device__ __forceinline__ float warp_inclusive_scan(float val) {
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        float other = __shfl_up_sync(0xFFFFFFFF, val, offset);
        if (threadIdx.x % 32 >= offset) val += other;
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

    // Use 256 threads (8 warps)
    extern __shared__ float sdata[]; // Array for warp totals
    float* warpSums = sdata;

    for (int col_start = 0; col_start < cols; col_start += blockDim.x) {
        int tid = threadIdx.x;
        int idx = col_start + tid;
        int warpId = tid >> 5;
        int lane   = tid & 31;

        // Load data
        float val = (idx < cols) ? row_in[idx] : 0.0f;

        // Warp-level scan
        val = warp_inclusive_scan(val);

        // Store warp sums
        if (lane == 31) {
            warpSums[warpId] = val;
        }
        __syncthreads();

        // Prefix sum across warps
        if (warpId > 0) {
            float prefix = 0.0f;
            for (int i = 0; i < warpId; ++i) {
                prefix += warpSums[i];
            }
            val += prefix;
        }

        // Apply carry from previous block
        if (col_start > 0) {
            val += row_out[col_start - 1];
        }

        // Write to global memory
        if (idx < cols) {
            row_out[idx] = val;
        }
        
        // Wait for all threads to write before proceeding to next tile
        __syncthreads();
    }
}

void fused_op_forward(int64_t rows, int64_t cols, torch::Tensor input, torch::Tensor output) {
    const int threads = 256;
    const int warps = threads >> 5;
    const int blocks = static_cast<int>(rows);
    // Shared memory stores only warp sums
    const size_t shmem = warps * sizeof(float);
    
    cumsum_kernel<<<blocks, threads, shmem>>>(
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
    m.def("fused_op", &fused_op_forward, "Fused cumsum execution via warp scan");
}
"""

# Compile the extension inline
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
    
    # Handle dimension: kernel assumes scan across last axis (cols)
    permute_dims = None
    if dim != -1 and dim != x.dim() - 1:
        permute_dims = list(range(x.dim()))
        permute_dims[dim], permute_dims[-1] = permute_dims[-1], permute_dims[dim]
        x = x.permute(*permute_dims)
    
    x = x.contiguous()
    output = torch.empty_like(x)
    
    rows = x.numel() // x.shape[-1]
    cols = x.shape[-1]
    
    fused_ext.fused_op(rows, cols, x, output)
    
    # Restore dimension order if needed
    if permute_dims is not None:
        output = output.permute(*permute_dims)
    
    return output.to(original_dtype)
