# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_152959/code_7.py
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

# ----------------------------------------------------------------------
# CUDA kernel: Max Reduction
# Uses cooperative block-wide reduction:
# 1. Thread-level partial max calculation.
# 2. Block-level reduction using shared memory.
# 3. __ldg for efficient cache-buffered reading.
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

template <typename scalar_t>
__global__ void reduce_max_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch,
    const int reduce_dim,
    const int other_dim,
    const int dim
) {
    extern __shared__ float sdata[];
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    
    // Each thread block processes one (batch, other_idx) tuple
    if (bid >= batch * other_dim) return;

    const int b = bid / other_dim;
    const int o = bid % other_dim;

    float local_max = -FLT_MAX;

    // Strided loading for efficiency
    if (dim == 1) {
        for (int i = tid; i < reduce_dim; i += blockDim.x) {
            float val = __ldg(input + b * reduce_dim * other_dim + i * other_dim + o);
            if (val > local_max) local_max = val;
        }
    } else {
        for (int i = tid; i < reduce_dim; i += blockDim.x) {
            float val = __ldg(input + b * reduce_dim * other_dim + o * reduce_dim + i);
            if (val > local_max) local_max = val;
        }
    }

    // Shared memory reduction
    sdata[tid] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdata[tid + s] > sdata[tid]) sdata[tid] = sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) output[bid] = sdata[0];
}

void max_reduce(
    torch::Tensor input,
    torch::Tensor output,
    const int dim
) {
    const int batch = input.size(0);
    const int dim1 = input.size(1);
    const int dim2 = input.size(2);
    
    const int reduce_dim = (dim == 1) ? dim1 : dim2;
    const int other_dim = (dim == 1) ? dim2 : dim1;
    const int n = batch * other_dim;
    
    const int threads = 256;
    const int blocks = n;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_reduce", ([&] {
        reduce_max_kernel<scalar_t><<<blocks, threads, threads * sizeof(float)>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch,
            reduce_dim,
            other_dim,
            dim
        );
    }));
}
"""

cpp_source = r"""
#include <torch/extension.h>

void max_reduce(torch::Tensor input, torch::Tensor output, const int dim);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("max_reduce", &max_reduce, "Max reduction along a dimension");
}
"""

# Compile the extension
max_reduce_ext = load_inline(
    name='max_reduce_cuda',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3'],
    with_cuda=True
)

def functional_model(x: torch.Tensor, *, dim: int) -> torch.Tensor:
    """
    Optimized max reduction using custom CUDA kernel.
    """
    if not x.is_cuda:
        x = x.cuda()
    
    batch = x.size(0)
    other_dim = x.size(2) if dim == 1 else x.size(1)
    
    out = torch.empty((batch, other_dim), dtype=x.dtype, device=x.device)
    
    max_reduce_ext.max_reduce(x, out, dim)
    return out
