# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_081907/code_22.py
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
# CUDA kernel – uses warp-level primitives to minimize synchronisation
# -------------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// Warp-level inclusive scan using shuffle instructions
// This avoids cross-thread blocks/syncthreads during the logic flow
__device__ __forceinline__ float warp_inclusive_scan(float val, int lane) {
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        float tmp = __shfl_up_sync(0xffffffff, val, offset);
        if (lane >= offset) val += tmp;
    }
    return val;
}

__global__ void cumsum_kernel(const float* __restrict__ input,
                              float* __restrict__ output,
                              int rows, int cols) {
    // Shared memory allocated dynamically: space for tiles + warp-sum buffer
    // 256 threads = 8 warps. We need 32 slots for warp sums (the max supported).
    extern __shared__ float sdata[];
    const int TILE_SIZE = blockDim.x;
    const int WARP_ID = threadIdx.x >> 5;
    const int LANE_ID = threadIdx.x & 31;

    int row = blockIdx.x;
    if (row >= rows) return;

    const float* row_in = input + row * cols;
    float* row_out = output + row * cols;

    float carry = 0.0f;

    // Process row in tiles of TILE_SIZE
    for (int col_start = 0; col_start < cols; col_start += TILE_SIZE) {
        int idx = col_start + threadIdx.x;

        // 1. Coalesced Load
        sdata[threadIdx.x] = (idx < cols) ? row_in[idx] : 0.0f;
        __syncthreads();

        // 2. Warp-level inclusive scan
        float val = sdata[threadIdx.x];
        val = warp_inclusive_scan(val, LANE_ID);

        // 3. Store warp inclusive sum in the warp-sum buffer section of shared memory
        if (LANE_ID == 31) {
            sdata[TILE_SIZE + WARP_ID] = val;
        }
        __syncthreads();

        // 4. Scan across warp sums (only first warp executes this)
        if (WARP_ID == 0) {
            float wval = (threadIdx.x < (TILE_SIZE >> 5)) ? sdata[TILE_SIZE + threadIdx.x] : 0.0f;
            wval = warp_inclusive_scan(wval, threadIdx.x);
            if (threadIdx.x < (TILE_SIZE >> 5)) sdata[TILE_SIZE + threadIdx.x] = wval;
        }
        __syncthreads();

        // 5. Add warp prefix
        if (WARP_ID > 0) {
            val += sdata[TILE_SIZE + WARP_ID - 1];
        }

        // 6. Add carry from previous tile
        val += carry;

        // Write result to local memory
        sdata[threadIdx.x] = val;
        __syncthreads();

        // Update carry for next tile
        carry = sdata[TILE_SIZE - 1];
        __syncthreads();

        // 7. Coalesced write to global
        if (idx < cols) {
            row_out[idx] = sdata[threadIdx.x];
        }
        __syncthreads();
    }
}

void fused_op_forward(int64_t rows, int64_t cols,
                      torch::Tensor input, torch::Tensor output) {
    const int threads = 256;
    const int blocks = static_cast<int>(rows);
    // Shared memory: TILE_SIZE elements for data + 32 elements for warp sums
    const size_t shared_mem = (threads + 32) * sizeof(float);

    cumsum_kernel<<<blocks, threads, shared_mem>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(rows),
        static_cast<int>(cols)
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_op_forward(int64_t rows, int64_t cols, torch::Tensor input, torch::Tensor output);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused cumsum execution");
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
    
    # Handle dimension: kernel assumes scan across the last dimension
    if dim != -1 and dim != x.dim() - 1:
        permute_dims = list(range(x.dim()))
        permute_dims[dim], permute_dims[-1] = permute_dims[-1], permute_dims[dim]
        x = x.permute(*permute_dims)
    
    x = x.contiguous()
    output = torch.empty_like(x)
    
    original_shape = x.shape
    rows = x.numel() // x.shape[-1]
    cols = x.shape[-1]
    
    # Invoke optimized kernel
    fused_ext.fused_op(rows, cols, x, output)
    
    # Restore dimension if perturbed
    if dim != -1 and dim != len(original_shape) - 1:
        output = output.view(original_shape)
        output = output.permute(*permute_dims)
    
    return output.to(original_dtype)
