# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_151757/code_7.py
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

# ------------------------------------------------------------
# CUDA kernel: tiled max reduction over the last dimension
# Optimized for coalesced global access and shared memory reduction
# ------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void max_last_dim_kernel(const float* __restrict__ input,
                                    float* output,
                                    const int N,
                                    const int D)
{
    // Total number of independent rows to reduce is N = B * D1
    // Each block handles one row of length D
    const int row = blockIdx.x;
    if (row >= N) return;

    extern __shared__ float sdata[];
    const int tid = threadIdx.x;
    
    float local_max = -3.4028235e38f; // -FLT_MAX
    const float* row_ptr = input + (size_t)row * D;

    // Grid-stride loop based on thread count to handle arbitrary D
    for (int j = tid; j < D; j += blockDim.x) {
        float val = row_ptr[j];
        if (val > local_max) local_max = val;
    }
    sdata[tid] = local_max;
    __syncthreads();

    // Tree-reduction inside shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdata[tid + s] > sdata[tid])
                sdata[tid] = sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) output[row] = sdata[0];
}

void max_last_dim(torch::Tensor input, torch::Tensor output) {
    const int N = input.numel() / input.size(-1);
    const int D = input.size(-1);
    
    // Use 256 threads per block
    const int threads = 256;
    const int blocks = N;

    max_last_dim_kernel<<<blocks, threads, threads * sizeof(float)>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N,
        D
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void max_last_dim(torch::Tensor input, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("max_last_dim", &max_last_dim, "Max reduction last dim");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='max_reduction',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True,
)

def functional_model(x, *, dim):
    """
    Optimized max reduction using custom CUDA kernel with tiled shared memory.
    """
    if not x.is_cuda:
        x = x.cuda()
    
    # Handle dimension permutations to always reduce last dimension
    if dim != x.ndim - 1:
        perm = list(range(x.ndim))
        perm.pop(dim)
        perm.append(dim)
        x = x.permute(perm).contiguous()
    
    # Prepare output shape
    out_shape = list(x.shape[:-1])
    out = torch.empty(out_shape, device=x.device, dtype=x.dtype)
    
    # Call kernel
    fused_ext.max_last_dim(x, out)
    
    # If we permuted, the original dim ordering for the non-reduced axes is preserved
    # by the output tensor logical shape, though we may need to revert 
    # if dim was explicitly not the last one and the logic requires specific ordering.
    # For this specific task (3D input), out shape is correct as is.
    return out
