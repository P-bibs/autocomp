# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_100811/code_5.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = []
FORWARD_ARG_NAMES = ['A', 'B']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = []
REQUIRED_FLAT_STATE_NAMES = []


class ModelNew(nn.Module):
    """
    Simple model that performs a single matrix multiplication (C = A * B) where one of the matrices is tall and skinny (M >> N or N >> M)
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

# CUDA kernel: Tiled MatMul. 
# We tile M (rows of A) and M (cols of B). 
# Since N is small (32), it fits entirely in shared memory.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

__global__ void matmul_tiled_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N) {
    // Shared memory for one tile of A (TILE_SIZE x N) and B (N x TILE_SIZE)
    __shared__ float sA[TILE_SIZE][32];
    __shared__ float sB[32][TILE_SIZE];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;

    // Iterate over the N dimension in blocks of 32
    // Since N=32, this loop runs once.
    for (int k_outer = 0; k_outer < N; k_outer += 32) {
        // Load into shared memory
        if (row < M && (k_outer + threadIdx.x) < N)
            sA[threadIdx.y][threadIdx.x] = A[row * N + (k_outer + threadIdx.x)];
        else
            sA[threadIdx.y][threadIdx.x] = 0.0f;

        if ((k_outer + threadIdx.y) < N && col < M)
            sB[threadIdx.y][threadIdx.x] = B[(k_outer + threadIdx.y) * M + col];
        else
            sB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < 32; ++k) {
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < M) {
        C[row * M + col] = sum;
    }
}

void fused_op_forward(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    int M = A.size(0);
    int N = A.size(1);
    
    // Grid dimensions: Use 32x32 blocks to maximize occupancy
    dim3 threads(32, 32);
    dim3 blocks((M + 31) / 32, (M + 31) / 32);
    
    matmul_tiled_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N
    );
}
"""

cpp_source = r"""
void fused_op_forward(torch::Tensor A, torch::Tensor B, torch::Tensor C);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Tiled Matrix Multiplication");
}
"""

fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(A, B):
    M = A.size(0)
    # Ensure C is initialized on the same device as inputs
    C = torch.empty((M, M), device=A.device, dtype=A.dtype)
    fused_ext.fused_op(A, B, C)
    return C

# --- Boilerplate ---
M = 16384 * 2
N = 16 * 2

def get_init_inputs():
    return []

def get_inputs():
    A = torch.rand(M, N, device='cuda', dtype=torch.float32)
    B = torch.rand(N, M, device='cuda', dtype=torch.float32)
    return [A, B]
