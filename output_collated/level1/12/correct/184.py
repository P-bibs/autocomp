# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_112116/code_27.py
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

# ----------------------------------------------------------------------
# CUDA kernel – uses a grid-stride loop (Optimization #7)
# This processes N*M elements efficiently, maximizing memory bandwidth.
# ----------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void broadcast_mul_kernel(
    const scalar_t* __restrict__ A,   // (N,)
    const scalar_t* __restrict__ B,   // (N, M)
    scalar_t* __restrict__ C,         // (N, M)
    const int64_t total_elements,
    const int64_t M)
{
    // Global thread id and stride for grid-stride loop
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = gridDim.x * blockDim.x;

    for (int64_t linear = idx; linear < total_elements; linear += stride) {
        // N index is linear / M. 
        // B[linear] and C[linear] are coalesced.
        // A[n] is shared across all columns M for a specific row.
        const int64_t n = linear / M;
        C[linear] = A[n] * B[linear];
    }
}

void broadcast_mul_forward(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    const int64_t N = A.size(0);
    const int64_t M = B.size(1);
    const int64_t total_elements = N * M;

    // Heuristic: 256 threads per block, calculate blocks needed
    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;
    
    // Cap blocks to prevent excessive hardware strain if necessary, 
    // though grid-stride handles any size.
    const int max_blocks = 65535;
    const int active_blocks = (blocks > max_blocks) ? max_blocks : blocks;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(A.scalar_type(), "broadcast_mul_forward", ([&] {
        broadcast_mul_kernel<scalar_t><<<active_blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            total_elements,
            M);
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}
"""

# ----------------------------------------------------------------------
# C++ binding
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void broadcast_mul_forward(torch::Tensor A, torch::Tensor B, torch::Tensor C);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("broadcast_mul", &broadcast_mul_forward, "Grid-stride broadcast multiply");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='broadcast_mul_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True,
    verbose=False
)

def functional_model(A, B):
    """
    Optimized version of A.unsqueeze(1) * B using a custom CUDA kernel 
    with a grid-stride loop.
    """
    # Ensure inputs are contiguous on GPU
    if not A.is_contiguous(): A = A.contiguous()
    if not B.is_contiguous(): B = B.contiguous()
    
    C = torch.empty_like(B)
    fused_ext.broadcast_mul(A, B, C)
    return C

def get_init_inputs():
    return []

def get_inputs():
    N, M = 4096, 4096
    A = torch.rand(N, device='cuda', dtype=torch.float32)
    B = torch.rand(N, M, device='cuda', dtype=torch.float32)
    return [A, B]
