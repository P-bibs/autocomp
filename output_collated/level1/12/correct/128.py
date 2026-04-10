# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_105940/code_27.py
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

# -----------------------------------------------------------------------------
# CUDA implementation optimized with Tiling and Memory Coalescing
# -----------------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// Define tile sizes for shared memory exploitation
#define TILE_M 32
#define TILE_N 16

template <typename scalar_t>
__global__ void broadcast_mul_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    const int N,
    const int M) 
{
    // Shared memory tile
    // We add 1 to the width dimension to pad shared memory and avoid bank conflicts
    __shared__ scalar_t tile[TILE_N][TILE_M + 1];

    int g_col = blockIdx.x * TILE_M + threadIdx.x;
    int g_row = blockIdx.y * TILE_N + threadIdx.y;

    // Load tile from global memory into shared memory
    if (g_row < N && g_col < M) {
        tile[threadIdx.y][threadIdx.x] = B[g_row * M + g_col];
    } else {
        tile[threadIdx.y][threadIdx.x] = 0;
    }

    __syncthreads();

    // Perform computation
    // Each thread multiplies its segment by A[g_row]
    if (g_row < N && g_col < M) {
        scalar_t a_val = A[g_row];
        C[g_row * M + g_col] = a_val * tile[threadIdx.y][threadIdx.x];
    }
}

void launch_broadcast_mul(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    const int N = A.size(0);
    const int M = B.size(1);

    dim3 threads(TILE_M, TILE_N);
    dim3 blocks((M + TILE_M - 1) / TILE_M, (N + TILE_N - 1) / TILE_N);

    AT_DISPATCH_FLOATING_TYPES(B.scalar_type(), "broadcast_mul", ([&] {
        broadcast_mul_kernel<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            N, M
        );
    }));
}
"""

cpp_source = r"""
#include <torch/extension.h>

void launch_broadcast_mul(torch::Tensor A, torch::Tensor B, torch::Tensor C);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("broadcast_mul", &launch_broadcast_mul, "Tiled Broadcast Multiply Kernel");
}
"""

# Compile the CUDA extension
fused_ext = load_inline(
    name='broadcast_mul_opt',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(A, B):
    """
    Optimized tiled broadcast multiply: C = A.unsqueeze(1) * B
    """
    # Ensure inputs are contiguous and on correct device
    A = A.contiguous()
    B = B.contiguous()
    C = torch.empty_like(B)
    
    fused_ext.broadcast_mul(A, B, C)
    return C

# -----------------------------------------------------------------------------
# Harness definitions as per requirements
# -----------------------------------------------------------------------------
M = 4096
N = 4096

def get_init_inputs():
    return []

def get_inputs():
    # Ensure tensors are created on CUDA as per performance environment
    A = torch.rand(N, device='cuda')
    B = torch.rand(N, M, device='cuda')
    return [A, B]
