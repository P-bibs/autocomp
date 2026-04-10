# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_094844/code_4.py
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

# --- CUDA Kernel ---
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_DIM 32

__global__ void tiled_matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N
) {
    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * TILE_DIM + ty;
    int col = bx * TILE_DIM + tx;

    float sum = 0.0f;

    // Loop over the tiles of the input matrices
    for (int k = 0; k < (N + TILE_DIM - 1) / TILE_DIM; ++k) {
        // Load A tile into shared memory
        if (row < M && (k * TILE_DIM + tx) < N)
            As[ty][tx] = A[row * N + (k * TILE_DIM + tx)];
        else
            As[ty][tx] = 0.0f;

        // Load B tile into shared memory
        if ((k * TILE_DIM + ty) < N && col < M)
            Bs[ty][tx] = B[(k * TILE_DIM + ty) * M + col];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();

        // Perform computation on the tile
        #pragma unroll
        for (int i = 0; i < TILE_DIM; ++i) {
            sum += As[ty][i] * Bs[i][tx];
        }

        __syncthreads();
    }

    if (row < M && col < M)
        C[row * M + col] = sum;
}

void tiled_matmul_forward(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& C) {
    int M = A.size(0);
    int N = A.size(1);
    
    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((M + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);

    tiled_matmul_kernel<<<grid, block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N
    );
    cudaDeviceSynchronize();
}
"""

# --- C++ Binding ---
cpp_source = r"""
void tiled_matmul_forward(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& C);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tiled_mm", &tiled_matmul_forward, "Tiled Matrix Multiplication");
}
"""

# Compile
tiled_mm_ext = load_inline(
    name='tiled_mm',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(A, B):
    M, N = A.shape
    C = torch.zeros((M, M), dtype=A.dtype, device=A.device)
    
    # Ensure contiguous for coalesced access
    tiled_mm_ext.tiled_mm(A.contiguous(), B.contiguous(), C)
    return C

def get_init_inputs():
    return []

def get_inputs():
    # M=32768, N=32
    M_val, N_val = 32768, 32
    A = torch.rand(M_val, N_val, device='cuda', dtype=torch.float32)
    B = torch.rand(N_val, M_val, device='cuda', dtype=torch.float32)
    return [A, B]
