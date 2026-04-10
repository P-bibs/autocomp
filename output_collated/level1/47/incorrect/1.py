# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_123707/code_9.py
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
# CUDA source – block‑parallel reduction using shared memory
# -------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void sum_dim1_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int B,
    const int D1,
    const int D2)
{
    // Flattened output index: (batch, k2) → batch*D2 + k2
    const int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= B * D2) return;

    const int batch = out_idx / D2;
    const int k2    = out_idx % D2;

    // Shared memory for the block‑wise reduction
    extern __shared__ float sdata[];

    // ---------- 1) each thread accumulates a partial sum ----------
    float sum = 0.0f;
    // strided access over the reduction dimension (D1)
    for (int i = threadIdx.x; i < D1; i += blockDim.x) {
        // row‑major index of input[batch, i, k2]
        int idx = ((batch * D1 + i) * D2) + k2;
        sum += input[idx];
    }
    sdata[threadIdx.x] = sum;
    __syncthreads();

    // ---------- 2) parallel reduction in shared memory ----------
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    // ---------- 3) write the final result ----------
    if (threadIdx.x == 0) {
        output[out_idx] = sdata[0];
    }
}

void sum_dim1_cuda(torch::Tensor input, torch::Tensor output) {
    const int B  = input.size(0);
    const int D1 = input.size(1);
    const int D2 = input.size(2);
    const int out_size = B * D2;               // B * D2  (keepdim → size 1 in dim 1)
    const int block_size = 256;
    const int grid_size  = (out_size + block_size - 1) / block_size;
    const int shared_mem = block_size * sizeof(float);

    sum_dim1_kernel<<<grid_size, block_size, shared_mem>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        B, D1, D2
    );
}
"""

# -------------------------------------------------------------------------
# C++ binding (PYBIND11) – exposes the CUDA routine to Python
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>
void sum_dim1_cuda(torch::Tensor input, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sum_dim1", &sum_dim1_cuda,
          "Block‑parallel sum over dimension 1 using shared memory");
}
"""

# Compile the extension (CUDA 12.5, optimization flags)
fused_ext = load_inline(
    name='sum_dim1_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# -------------------------------------------------------------------------
# The only function that will be imported – optimized functional_model
# -------------------------------------------------------------------------
def functional_model(x, *, dim):
    """
    Sum of tensor `x` over the dimension `dim` with `keepdim=True`.
    Only `dim=1` (the second axis) is implemented with a custom CUDA kernel.
    """
    if dim != 1:
        # The original code always uses dim=1, so this path is never hit.
        raise NotImplementedError("Custom kernel supports only dim=1")

    # Ensure a contiguous float32 view for the kernel
    orig_dtype = x.dtype
    if x.dtype != torch.float32:
        x = x.float()

    if not x.is_contiguous():
        x = x.contiguous()

    B, D1, D2 = x.shape               # (128, 4096, 4095)

    # Allocate output tensor with shape (B, 1, D2) – keepdim=True
    output = torch.empty((B, 1, D2), dtype=x.dtype, device=x.device)

    # Flatten output to a 1‑D tensor of size B*D2 (row‑major order)
    out_flat = output.view(-1)        # shape = (B * D2)

    # Run the custom reduction kernel
    fused_ext.sum_dim1(x, out_flat)

    # Cast back to the original dtype if necessary
    if orig_dtype != torch.float32:
        output = output.to(dtype=orig_dtype)

    return output
