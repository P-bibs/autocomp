# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_102701/code_26.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = []
FORWARD_ARG_NAMES = ['A', 'B']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = []
REQUIRED_FLAT_STATE_NAMES = []


class ModelNew(nn.Module):
    """
    Simple model that performs a matrix multiplication of a diagonal matrix with another matrix.
    C = diag(A) * B
    """

    def __init__(self):
        super(ModelNew, self).__init__()

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

# ------------------------------------------------------------------
# CUDA kernel: element‑wise multiply A (N,) with each row of B (N,M)
# Each thread calculates one element of the output matrix.
# ------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void fused_mul_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ out,
    const int N,
    const int M)
{
    // Calculate global thread index
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = static_cast<size_t>(N) * static_cast<size_t>(M);

    // Grid-stride loop handles arbitrary input sizes
    for (; idx < total; idx += blockDim.x * gridDim.x) {
        int i = idx / M; // Row index corresponding to A
        out[idx] = A[i] * B[idx];
    }
}

void fused_mul_forward(torch::Tensor A, torch::Tensor B, torch::Tensor out) {
    const int N = A.size(0);
    const int M = B.size(1);
    const size_t total = static_cast<size_t>(N) * static_cast<size_t>(M);
    
    // Choose block/grid dimensions
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "fused_mul_kernel", ([&] {
        fused_mul_kernel<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            N,
            M
        );
    }));
}
"""

# ------------------------------------------------------------------
# C++ Binding
# ------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>
void fused_mul_forward(torch::Tensor A, torch::Tensor B, torch::Tensor out);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_mul", &fused_mul_forward, "Fused A * B broadcasted multiplication");
}
"""

# ------------------------------------------------------------------
# Compile the module
# ------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_mul_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True,
    verbose=False
)

def functional_model(A, B):
    """
    Optimized implementation of A.unsqueeze(1) * B using a custom CUDA kernel.
    Ensures optimal memory layout and fused computation.
    """
    # Ensure inputs are on CUDA and contiguous
    device = torch.device('cuda')
    A = A.to(device).contiguous()
    B = B.to(device).contiguous()
    
    out = torch.empty_like(B)
    fused_ext.fused_mul(A, B, out)
    return out

# ------------------------------------------------------------------
# Boilerplate for evaluation
# ------------------------------------------------------------------
M = 4096
N = 4096

def get_init_inputs():
    return []

def get_inputs():
    A = torch.rand(N, dtype=torch.float32)
    B = torch.rand(N, M, dtype=torch.float32)
    return [A, B]
