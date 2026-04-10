# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_121327/code_3.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['dim']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['dim']
REQUIRED_FLAT_STATE_NAMES = []


class ModelNew(nn.Module):
    """
    Simple model that performs sum reduction over a specified dimension.
    """

    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): Dimension to reduce over.
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
# 1. CUDA kernel (device code)
# -------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// Warp-level reduction using __shfl_xor_sync (builtin primitive)
__device__ __forceinline__ float warp_reduce(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

// Kernel: reduces dimension 1 (dim1) for each (batch, col) pair.
// grid = batch * dim2  → one block per output element
// blockDim = 256 (8 warps)
__global__ void reduce_sum_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int batch,
    const int dim1,
    const int dim2)
{
    // linear index of the output element (batch * dim2 + col)
    const int out_idx = blockIdx.x;

    const int batch_idx = out_idx / dim2;
    const int col_idx   = out_idx % dim2;

    // pointer to the start of the reduction vector for this (batch, col)
    const float* base = input + batch_idx * dim1 * dim2 + col_idx;

    // ---------------------------------------------------------
    // 1) Partial sum in registers (strided loads)
    // ---------------------------------------------------------
    float sum = 0.0f;
    for (int i = threadIdx.x; i < dim1; i += blockDim.x) {
        sum += base[i * dim2];               // coalesced across the stride
    }

    // ---------------------------------------------------------
    // 2) First warp-level reduction (within each warp)
    // ---------------------------------------------------------
    sum = warp_reduce(sum);

    // ---------------------------------------------------------
    // 3) Store each warp's result to shared memory
    // ---------------------------------------------------------
    __shared__ float sdata[32];               // max 256 threads → 8 warps
    const int lane = threadIdx.x & 0x1f;      // lane index inside warp
    const int warp_id = threadIdx.x >> 5;     // warp index

    if (lane == 0) {
        sdata[warp_id] = sum;                // one value per warp
    }
    __syncthreads();

    // ---------------------------------------------------------
    // 4) Final warp reduction across the warp-summaries
    // ---------------------------------------------------------
    if (threadIdx.x < blockDim.x / 32) {
        sum = sdata[threadIdx.x];
        sum = warp_reduce(sum);
    }

    // ---------------------------------------------------------
    // 5) Write final result (only thread 0 of the block needed)
    // ---------------------------------------------------------
    if (threadIdx.x == 0) {
        output[out_idx] = sum;
    }
}

// Host wrapper that launches the kernel
void fused_op_forward(
    torch::Tensor input,   // (batch, dim1, dim2) contiguous float tensor
    torch::Tensor output,  // (batch, 1, dim2) contiguous float tensor
    int batch,
    int dim1,
    int dim2)
{
    const int threads = 256;
    const int blocks  = batch * dim2;   // one block per output element

    reduce_sum_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch, dim1, dim2);
}
"""

# -------------------------------------------------------------------------
# 2. C++ binding (PYBIND11)
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(
    torch::Tensor input,
    torch::Tensor output,
    int batch,
    int dim1,
    int dim2);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward,
          "Fused sum reduction over dim=1 using a custom CUDA kernel");
}
"""

# -------------------------------------------------------------------------
# 3. Build the inline extension
# -------------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_reduction',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True,
)

# -------------------------------------------------------------------------
# 4. Optimized functional_model (the only symbol imported for evaluation)
# -------------------------------------------------------------------------
def functional_model(x, *, dim):
    """
    Custom sum over the specified dimension.
    Only dim==1 is optimized; any other value falls back to torch.sum.
    """
    # -----------------------------------------------------------------
    # The benchmark always reduces over dim==1, so the fallback is only
    # a safety net.  In practice we never hit it.
    # -----------------------------------------------------------------
    if dim != 1:
        # Fallback for completeness (not used in the benchmark)
        return torch.sum(x, dim=dim, keepdim=True)

    # The kernel expects a contiguous float32 tensor on the GPU.
    if not x.is_cuda:
        raise RuntimeError("Input tensor must be on GPU")
    if x.dtype != torch.float32:
        x = x.to(torch.float32)          # promote to float32
    if not x.is_contiguous():
        x = x.contiguous()

    batch = x.size(0)
    dim1  = x.size(1)
    dim2  = x.size(2)

    # Output shape: (batch, 1, dim2)   → keepdim=True
    out = torch.empty((batch, 1, dim2), dtype=x.dtype, device=x.device)

    # Launch the hand-tuned CUDA reduction
    fused_ext.fused_op(x, out, batch, dim1, dim2)

    return out

batch_size = 128
dim1 = 4096
dim2 = 4095
reduce_dim = 1

def get_init_inputs():
    return [reduce_dim]

def get_inputs():
    x = torch.rand(batch_size, dim1, dim2)
    return [x]
