# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_123707/code_18.py
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

# CUDA kernel code
# We perform a reduction over the middle dimension (D).
# The data is layout [B, D, N]. To access elements contiguously as much as possible,
# we process (b, n) pairs. The reduction accesses x[b, d, n] where d varies.
# This means for a fixed (b, n), we jump by N*sizeof(type) in memory.
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void sum_dim1_kernel(
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ y,
    const int B,
    const int D,
    const int N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = B * N;

    for (int i = idx; i < total_threads; i += blockDim.x * gridDim.x) {
        const int b = i / N;
        const int n = i % N;
        
        scalar_t sum = scalar_t(0);
        // Pointer to the start of the reduction column: x[b, 0, n]
        const scalar_t* data_ptr = x + b * D * N + n;
        
        #pragma unroll
        for (int d = 0; d < D; ++d) {
            sum += data_ptr[d * N];
        }
        y[b * N + n] = sum;
    }
}

void fused_sum_forward(torch::Tensor x, torch::Tensor y) {
    const int B = (int)x.size(0);
    const int D = (int)x.size(1);
    const int N = (int)x.size(2);

    const int threads = 256;
    const int blocks = (B * N + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "fused_sum_forward", ([&] {
        sum_dim1_kernel<scalar_t><<<blocks, threads>>>(
            x.data_ptr<scalar_t>(),
            y.data_ptr<scalar_t>(),
            B, D, N
        );
    }));
}
"""

# C++ binding code
cpp_source = r"""
#include <torch/extension.h>

void fused_sum_forward(torch::Tensor x, torch::Tensor y);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_sum", &fused_sum_forward, "Fused reduction along dim 1");
}
"""

# Compile the inline extension
fused_sum_module = load_inline(
    name="fused_sum_ext",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True,
)

def functional_model(x, *, dim):
    # Requirements: input [128, 4096, 4095], reduce dim 1, keepdim=True
    # Output shape should be [128, 1, 4095]
    batch_size = x.size(0)
    inner_dim = x.size(2)
    
    out = torch.empty(batch_size, 1, inner_dim, dtype=x.dtype, device=x.device)
    fused_sum_module.fused_sum(x, out)
    return out

# Constants to match original challenge parameters
batch_size = 128
dim1 = 4096
dim2 = 4095
reduce_dim = 1

def get_init_inputs():
    return [reduce_dim]

def get_inputs():
    x = torch.rand(batch_size, dim1, dim2, device='cuda')
    return [x]
