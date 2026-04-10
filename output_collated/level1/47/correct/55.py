# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_131936/code_27.py
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
Optimized implementation of functional_model – reduction over dim 1.
The kernel utilizes a fully coalesced memory access pattern where each thread 
is responsible for one column index 'j', effectively streaming memory for all 
elements in the D1 dimension.
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

// Kernel performs a sum reduction over dimension 1 (D1)
// Input: (B, D1, D2), Output: (B, D2)
__global__ void sum_dim1_kernel(const float* __restrict__ input,
                                float* __restrict__ output,
                                const int B,
                                const int D1,
                                const int D2)
{
    const int b = blockIdx.x;
    const int j = blockIdx.y * blockDim.x + threadIdx.x;

    if (j >= D2) return;

    // Use float for accumulation; coalesced access achieved as consecutive threads
    // access consecutive memory addresses in the D2 dimension.
    float sum = 0.0f;
    const size_t batch_offset = static_cast<size_t>(b) * D1 * D2;
    
    // Each thread loops through the D1 dimension for its specific column j
    for (int i = 0; i < D1; ++i)
    {
        sum += input[batch_offset + i * D2 + j];
    }

    output[static_cast<size_t>(b) * D2 + j] = sum;
}

void sum_dim1(torch::Tensor input, torch::Tensor output)
{
    const int B  = input.size(0);
    const int D1 = input.size(1);
    const int D2 = input.size(2);

    // 256 threads per block is optimal for occupancy on most architectures
    const int threads = 256;
    dim3 grid(B, DIV_UP(D2, threads));

    sum_dim1_kernel<<<grid, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        B, D1, D2
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void sum_dim1(torch::Tensor input, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sum_dim1", &sum_dim1, "Sum along dimension 1 with coalesced accesses");
}
"""

# Compile the inline extension
sum_ext = load_inline(
    name='sum_dim1',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True,
    verbose=False
)

def functional_model(x, *, dim):
    """
    Optimized reduction over dimension 1.
    Maintains semantic compatibility with the original (B, 1, D2) output shape.
    """
    assert dim == 1
    B, D1, D2 = x.shape
    # Pre-allocate output with the internal shape (B, D2)
    output = torch.empty((B, D2), device=x.device, dtype=x.dtype)
    
    # Call the optimized CUDA kernel
    sum_ext.sum_dim1(x, output)
    
    # Unsqueeze to match the (B, 1, D2) requirement
    return output.unsqueeze(1)

# --- Evaluation setup ---
batch_size = 128
dim1 = 4096
dim2 = 4095
reduce_dim = 1

def get_init_inputs():
    return [reduce_dim]

def get_inputs():
    x = torch.rand(batch_size, dim1, dim2, device='cuda', dtype=torch.float32)
    return [x]
