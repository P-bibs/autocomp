# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_131936/code_11.py
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




# -*- coding: utf-8 -*-
"""
Optimised implementation of functional_model – reduction over dim 1.
The optimisation applied:
    2. Coalesce global memory accesses.
The kernel now lets each thread handle a single column index `j`,
producing fully coalesced loads while iterating over the reduction
dimension `i`.  No vectorisation or extra branching is required.
"""

import torch
from torch.utils.cpp_extension import load_inline

# --------------------------------------------------------------
#  CUDA kernel – fully coalesced reads
# --------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#ifndef DIV_UP
#define DIV_UP(a,b) (((a)+(b)-1)/(b))
#endif

// ----------------------------------------------------------
//  sum_dim1_kernel
//  Input  : (B, D1, D2)
//  Output : (B, D2)   – sum over dimension 1 (the D1 axis)
// ----------------------------------------------------------
__global__ void sum_dim1_kernel(const float* __restrict__ input,
                                float* __restrict__ output,
                                const int B,
                                const int D1,
                                const int D2)
{
    //  blockIdx.x  -> batch element
    //  blockIdx.y  -> tile of columns (j)
    //  threadIdx.x -> column inside the tile
    const int b = blockIdx.x;
    const int tile_start = blockIdx.y * blockDim.x;          // first column handled by this block
    const int j = tile_start + threadIdx.x;                 // column index for this thread

    if (b >= B || j >= D2) return;   // out‑of‑range guard (very cheap)

    // ------------------------------------------------------------------
    //  Reduce over the D1 dimension.
    //  For a fixed (b, j) the address pattern is:
    //      offset = b*D1*D2 + i*D2 + j   (i = 0 … D1‑1)
    //  Consecutive threads differ only in `j`, therefore each iteration
    //  over i yields fully coalesced loads.
    // ------------------------------------------------------------------
    float sum = 0.0f;
    const size_t base = static_cast<size_t>(b) * D1 * D2;   // start of this batch element

    // Grid‑stride loop over i (optional, but useful when D1 is huge)
    for (int i = 0; i < D1; ++i)
    {
        sum += input[base + i * D2 + j];
    }

    // Store the result
    output[static_cast<size_t>(b) * D2 + j] = sum;
}

// ------------------------------------------------------------------
//  C++ wrapper (launches the kernel)
// ------------------------------------------------------------------
void sum_dim1(torch::Tensor input, torch::Tensor output)
{
    // sanity checks – they are cheap and help catch user errors early
    TORCH_CHECK(input.dim() == 3, "input must be 3‑D");
    TORCH_CHECK(output.dim() == 2, "output must be 2‑D");
    TORCH_CHECK(input.is_cuda() && output.is_cuda(),
                "both tensors must be on CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat && output.dtype() == torch::kFloat,
                "only float32 is supported");

    const int B  = input.size(0);
    const int D1 = input.size(1);
    const int D2 = input.size(2);

    // --------------------------------------------------------------
    //  Kernel launch configuration
    //  One thread per output column (j).  We tile the columns with
    //  blockDim.x = 256 (fits nicely on an RTX 2080 Ti) and let the y‑grid
    //  cover the whole D2 dimension.
    // --------------------------------------------------------------
    const int threads = 256;
    const dim3 threads_per_block(threads);
    const dim3 blocks(B, DIV_UP(D2, threads));

    sum_dim1_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        B, D1, D2
    );
    // Propagate CUDA errors (useful for debugging)
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("CUDA error in sum_dim1 kernel: ", cudaGetErrorString(err));
    }
}
"""

# --------------------------------------------------------------
#  C++ binding (PYBIND11)
# --------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

// Forward declaration of the kernel launcher defined in the .cu file
void sum_dim1(torch::Tensor input, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sum_dim1", &sum_dim1,
          "Sum along dimension 1 (CUDA, fully coalesced loads)");
}
"""

# Compile the inline extension
sum_ext = load_inline(
    name="sum_dim1",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    with_cuda=True,
    verbose=False,
)

# ------------------------------------------------------------------
#  functional_model – public entry point used by the benchmark harness
# ------------------------------------------------------------------
def functional_model(x, *, dim):
    """
    Reduce a tensor of shape (B, D1, D2) over dim = 1.
    Returns a tensor of shape (B, 1, D2) (identical to the original API).
    """
    assert dim == 1, "Only dim=1 is supported by this specialised kernel"
    B, D1, D2 = x.shape
    # Allocate output (B, D2) – same device / dtype as the input
    out = torch.empty((B, D2), device=x.device, dtype=x.dtype)
    # Launch the custom CUDA kernel
    sum_ext.sum_dim1(x, out)
    # Match the original shape (B, 1, D2)
    return out.unsqueeze(1)

# ------------------------------------------------------------------
#  Helper functions used by the evaluation script (unchanged)
# ------------------------------------------------------------------
batch_size = 128
dim1 = 4096
dim2 = 4095
reduce_dim = 1

def get_init_inputs():
    return [reduce_dim]

def get_inputs():
    x = torch.rand(batch_size, dim1, dim2, device="cuda", dtype=torch.float32)
    return [x]
