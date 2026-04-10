# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_143434/code_15.py
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
# CUDA kernel source – reduction using warp‑level primitives
# -------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

constexpr int BLOCK_SIZE = 256;   // chosen for good occupancy on RTX 2080 Ti

// Kernel: one block per (batch, dim2) output element.
// Each block reduces dim1 (=4096) elements using a tree in shared memory,
// then finishes with a warp‑shuffle reduction.
__global__ void sum_reduce_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int batch,
    const int dim1,
    const int dim2)
{
    // Linear index of the output element: (b, d2)
    int idx = blockIdx.x;
    if (idx >= batch * dim2) return;
    int b = idx / dim2;
    int d2 = idx % dim2;

    // Pointer to the slice that must be summed: x[b, :, d2]
    const float* slice = input + (b * dim1 * dim2 + d2 * dim1);

    // Shared memory for the block‑wise partial sums
    __shared__ float sdata[BLOCK_SIZE];

    int tid = threadIdx.x;

    // Each thread loads a strided chunk of the slice (coalesced access)
    float sum = 0.0f;
    for (int i = tid; i < dim1; i += BLOCK_SIZE) {
        sum += slice[i];
    }
    sdata[tid] = sum;
    __syncthreads();

    // Block‑level reduction (power‑of‑two steps, > 32 threads)
    for (int s = BLOCK_SIZE / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Warp‑level reduction using __shfl_down_sync (no extra syncthreads)
    if (tid < 32) {
        float val = sdata[tid];
        if (BLOCK_SIZE >= 64) val += sdata[tid + 32];
        // Intra‑warp tree via shuffle
        val += __shfl_down_sync(0xffffffff, val, 16);
        val += __shfl_down_sync(0xffffffff, val, 8);
        val += __shfl_down_sync(0xffffffff, val, 4);
        val += __shfl_down_sync(0xffffffff, val, 2);
        val += __shfl_down_sync(0xffffffff, val, 1);
        sdata[tid] = val;
    }
    __syncthreads();

    // Write the final reduced value
    if (tid == 0) {
        output[idx] = sdata[0];
    }
}

// Host wrapper that launches the kernel
void fused_op_forward(
    const torch::Tensor& input,
    torch::Tensor& output,
    int batch,
    int dim1,
    int dim2)
{
    const int blocks = batch * dim2;      // one block per output element
    const int threads = BLOCK_SIZE;       // 256 threads per block
    sum_reduce_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch, dim1, dim2);
    cudaDeviceSynchronize();
}
"""

# -------------------------------------------------------------------------
# C++/pybind11 interface – exposes fused_op to Python
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(
    const torch::Tensor& input,
    torch::Tensor& output,
    int batch,
    int dim1,
    int dim2);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward,
          "Custom sum reduction along dim=1 using warp‑level primitives");
}
"""

# -------------------------------------------------------------------------
# Build the inline CUDA extension with -O3 and --use_fast_math
# -------------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# -------------------------------------------------------------------------
# functional_model – the only function imported for evaluation
# -------------------------------------------------------------------------
def functional_model(x, *, dim):
    """
    Optimised reduction along the second dimension (dim=1) using a custom
    CUDA kernel that leverages warp‑level shuffle instructions.
    """
    # Input validation
    if not x.is_cuda:
        raise RuntimeError("Input tensor must be on CUDA")
    if x.dim() != 3:
        raise RuntimeError("Input must be a 3-D tensor")
    batch, dim1, dim2 = x.shape
    if dim != 1:
        raise RuntimeError("This optimized implementation only supports dim=1")

    # Output shape: (batch, 1, dim2) – keepdim=True
    output = torch.empty((batch, 1, dim2), dtype=x.dtype, device=x.device)

    # Flatten the tensors for the kernel (row‑major layout)
    out_flat = output.view(-1)          # size = batch * dim2
    x_flat   = x.view(-1)               # size = batch * dim1 * dim2

    # Launch the custom kernel
    fused_ext.fused_op(x_flat, out_flat, batch, dim1, dim2)

    return output

batch_size = 128
dim1 = 4096
dim2 = 4095
reduce_dim = 1

def get_init_inputs():
    return [reduce_dim]

def get_inputs():
    x = torch.rand(batch_size, dim1, dim2, device='cuda')
    return [x]
