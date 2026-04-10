# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_151147/code_3.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['dim']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['dim']
REQUIRED_FLAT_STATE_NAMES = []


class ModelNew(nn.Module):
    """
    Simple model that performs Max reduction over a specific dimension.
    """

    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): The dimension to reduce over.
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
# Inline CUDA kernel (max reduction along the last dimension)
# -------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>   // for FLT_MAX

// Maximum reduction kernel: each block reduces one "row" of length row_size.
// The grid is 1D: one block per output element.
__global__ void max_reduce_kernel(const float* __restrict__ input,
                                   float* output,
                                   int row_size,
                                   int num_rows)
{
    // Shared memory for per‑warp results (max 8 warps for blockDim=256)
    __shared__ float sdata[256];

    int tid = threadIdx.x;
    int row = blockIdx.x;                 // which output element this block handles
    if (row >= num_rows) return;

    const float* row_ptr = input + row * row_size;

    // ---- first loop: each thread keeps its own running maximum ----
    float val = -FLT_MAX;                 // start with negative infinity
    for (int i = tid; i < row_size; i += blockDim.x) {
        float v = row_ptr[i];
        if (v > val) val = v;
    }

    // ---- warp‑level reduction (no sync needed inside a warp) ----
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(0xffffffff, val, offset);
        if (other > val) val = other;
    }

    // ---- store warp result to shared memory ----
    if (tid % warpSize == 0) {
        sdata[tid / warpSize] = val;
    }
    __syncthreads();

    // ---- final warp reduction across the warps in the block ----
    // Number of warps per block = blockDim.x / warpSize = 8
    if (tid < blockDim.x / warpSize) {
        val = sdata[tid];
    } else {
        val = -FLT_MAX;
    }
    for (int offset = (blockDim.x / warpSize) / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            float other = __shfl_down_sync(0xffffffff, val, offset);
            if (other > val) val = other;
        }
    }

    // ---- write the final maximum for this row ----
    if (tid == 0) {
        output[row] = val;
    }
}

// Launch wrapper callable from Python
void max_reduce(const torch::Tensor &input, torch::Tensor &output) {
    // Assuming input is contiguous and float32, shape (N, M)
    const int row_size = input.size(1);
    const int num_rows = input.size(0);

    const int threads = 256;                 // block size, multiple of 32
    const int blocks  = num_rows;            // one block per output element

    max_reduce_kernel<<<blocks, threads, 0, cudaStreamDefault>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        row_size,
        num_rows
    );

    // Optional: cudaDeviceSynchronize() is not needed because the kernel
    // is launched with the default stream and we will sync before reading
    // the output on the host.
    cudaDeviceSynchronize();
}
"""

# -------------------------------------------------------------------------
# C++ binding (PYBIND11)
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void max_reduce(const torch::Tensor &input, torch::Tensor &output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("max_reduce", &max_reduce, "max reduction along last dimension");
}
"""

# -------------------------------------------------------------------------
# Compile the inline extension
# -------------------------------------------------------------------------
fused_ext = load_inline(
    name='max_reduce_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True,
)

# -------------------------------------------------------------------------
# Helper functions required by the benchmark harness
# -------------------------------------------------------------------------
def get_init_inputs():
    # Not used by functional_model, but kept for completeness
    return [1]

def get_inputs():
    batch_size = 128
    dim1 = 4096
    dim2 = 4095
    x = torch.rand(batch_size, dim1, dim2)
    return [x]

# -------------------------------------------------------------------------
# The function to be evaluated
# -------------------------------------------------------------------------
def functional_model(x, *, dim):
    """
    Returns the maximum of `x` along the given `dim`.
    The implementation uses a hand‑written CUDA kernel to achieve
    high throughput on the RTX 2080 Ti.
    """
    # Ensure we work on a contiguous float tensor
    orig_dtype = x.dtype
    if x.dtype != torch.float32:
        x = x.to(torch.float32)
    x = x.contiguous()

    ndim = x.dim()
    # Normalise negative dimension
    if dim < 0:
        dim = ndim + dim

    # If the reduction dimension is not the last one, permute it to the end
    if dim != ndim - 1:
        perm = list(range(ndim))
        perm[dim], perm[-1] = perm[-1], perm[dim]
        x = x.permute(perm).contiguous()

    # Flatten leading dimensions: (N, M) where M is the reduction size
    shape = x.shape
    M = shape[-1]                     # size of the reduced dimension
    N = 1
    for s in shape[:-1]:
        N *= s

    x = x.view(N, M)                  # (N, M) contiguous view

    # Allocate output tensor
    output = torch.empty(N, dtype=torch.float32, device='cuda')

    # Invoke the custom CUDA reduction
    fused_ext.max_reduce(x, output)

    # Reshape back to the original shape without the reduced dimension
    out_shape = list(shape[:-1])      # product of leading dims
    output = output.view(out_shape)

    # Convert back to the original dtype if necessary
    if orig_dtype != torch.float32:
        output = output.to(orig_dtype)

    return output
