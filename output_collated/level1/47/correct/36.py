# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_125421/code_28.py
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
# CUDA source – kernel + host function that launches it
# -------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// Each block computes one (b, j) coordinate
__global__ void reduce_sum_kernel(const float* __restrict__ input,
                                   float* __restrict__ output,
                                   const int batch,
                                   const int dim1,
                                   const int dim2)
{
    int b = blockIdx.x / dim2;
    int j = blockIdx.x % dim2;

    // Pointer to this (b, :, j) slice
    const float* ptr = input + (b * dim1 * dim2) + j;

    // Use private register variable to accumulate
    float sum = 0.0f;
    for (int i = threadIdx.x; i < dim1; i += blockDim.x) {
        sum += ptr[i * dim2];
    }

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Shared memory for warp results (max 32 warps for 1024 threads)
    __shared__ float warp_sums[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    if (lane == 0) {
        warp_sums[wid] = sum;
    }
    __syncthreads();

    // Final reduction
    if (wid == 0) {
        float final_val = (threadIdx.x < (blockDim.x / 32)) ? warp_sums[threadIdx.x] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            final_val += __shfl_down_sync(0xffffffff, final_val, offset);
        }
        if (threadIdx.x == 0) {
            output[b * dim2 + j] = final_val;
        }
    }
}

void reduce_sum_launcher(torch::Tensor input, torch::Tensor output,
                         int batch, int dim1, int dim2) {
    // 256 threads is generally optimal for latency hiding in reductions
    int threads = 256;
    int blocks = batch * dim2;
    
    reduce_sum_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch, dim1, dim2
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void reduce_sum_launcher(torch::Tensor input, torch::Tensor output,
                         int batch, int dim1, int dim2);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("reduce_sum", &reduce_sum_launcher, "Warp-level reduction sum");
}
"""

# Compile extension once
module = load_inline(
    name='reduce_sum_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, dim):
    if dim != 1:
        raise ValueError("Only dim=1 supported")
    
    batch, d1, d2 = x.shape
    # Output buffer
    out = torch.empty((batch, d2), device=x.device, dtype=x.dtype)
    
    module.reduce_sum(x.contiguous(), out, batch, d1, d2)
    
    # Reshape to (batch, 1, dim2) as per requirement
    return out.view(batch, 1, d2)
