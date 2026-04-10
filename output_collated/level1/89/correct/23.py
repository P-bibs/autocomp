# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_080721/code_10.py
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
# CUDA kernel – warp-level scan with 256 threads per block
# ----------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void cumsum_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows,
    int cols)
{
    // Shared memory: tile data (threads) + per-warp sums (warps)
    extern __shared__ float sdata[];
    const int threads = blockDim.x;
    const int warps   = threads / 32;

    const int row = blockIdx.x;
    if (row >= rows) return;

    const float* row_in  = input  + row * cols;
    float*       row_out = output + row * cols;

    // ------------------------------------------------------------------
    // Tiled scan – each tile has 'threads' elements
    // ------------------------------------------------------------------
    for (int col_start = 0; col_start < cols; col_start += threads) {
        int tid = threadIdx.x;
        int idx = col_start + tid;

        // ---- Coalesced load ------------------------------------------------
        sdata[tid] = (idx < cols) ? row_in[idx] : 0.0f;
        __syncthreads();

        // ---- Warp-level inclusive scan (Kogge-Stone using shuffle) ----------
        int lane = tid & 31;          // lane index inside warp
        int warp = tid >> 5;          // warp index inside block

        float val = sdata[tid];
        for (int offset = 1; offset < 32; offset <<= 1) {
            float tmp = __shfl_up_sync(0xffffffff, val, offset);
            if (lane >= offset) val += tmp;
        }
        // now 'val' holds the inclusive scan inside the warp
        sdata[tid] = val;

        // ---- Store warp totals (inclusive) ---------------------------------
        if (lane == 31) {
            sdata[threads + warp] = val;      // warp_sums[warp]
        }
        __syncthreads();

        // ---- Prefix-sum across warp totals (simple serial scan in thread 0) --
        if (tid == 0) {
            for (int i = 1; i < warps; ++i) {
                sdata[threads + i] += sdata[threads + i - 1];
            }
        }
        __syncthreads();

        // ---- Add carry from previous warps ---------------------------------
        if (warp > 0) {
            sdata[tid] += sdata[threads + warp - 1];
        }
        __syncthreads();

        // ---- Add carry from previous tile (global) -------------------------
        if (col_start > 0) {
            sdata[tid] += row_out[col_start - 1];
        }
        __syncthreads();

        // ---- Coalesced store ------------------------------------------------
        if (idx < cols) {
            row_out[idx] = sdata[tid];
        }
        __syncthreads();
    }
}

// Host side launch routine
void fused_op_forward(int64_t rows, int64_t cols,
                      torch::Tensor input, torch::Tensor output)
{
    const int threads = 256;                     // multiple of 32 → better occupancy
    const int blocks  = static_cast<int>(rows);
    const int warps   = threads / 32;
    const size_t shared_mem = (threads + warps) * sizeof(float);

    cumsum_kernel<<<blocks, threads, shared_mem>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(rows),
        static_cast<int>(cols)
    );
}
"""

# ----------------------------------------------------------------------
# C++ binding (PyBind11)
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>
void fused_op_forward(int64_t rows, int64_t cols,
                      torch::Tensor input, torch::Tensor output);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused inclusive scan (cumsum) on the last dim");
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
# Public API – same signature as the original functional_model
# ----------------------------------------------------------------------
def functional_model(x, *, dim):
    # Keep the original dtype and force FP32 for the kernel
    original_dtype = x.dtype
    x = x.to(torch.float32)

    # Permute so that the scan is always along the last dimension
    if dim != -1 and dim != x.dim() - 1:
        perm = list(range(x.dim()))
        perm[dim], perm[-1] = perm[-1], perm[dim]
        x = x.permute(*perm)

    x = x.contiguous()
    output = torch.empty_like(x)

    # Prepare launch parameters
    rows = x.numel() // x.shape[-1]
    cols = x.shape[-1]

    # Call the optimized CUDA kernel
    fused_ext.fused_op(rows, cols, x, output)

    # Restore original dimension ordering if needed
    if dim != -1 and dim != len(x.shape) - 1:
        output = output.view(x.shape)
        output = output.permute(*perm)

    return output.to(original_dtype)
