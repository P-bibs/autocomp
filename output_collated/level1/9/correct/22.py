# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_100346/code_3.py
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

# -------------------------------------------------------------------------
# Sizes used by the original benchmark
M = 16384 * 2        # 32768
N = 16 * 2           # 32

# -------------------------------------------------------------------------
# CUDA source – tiled matrix‑multiply kernel using shared memory
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

__global__ void tiled_matmul_kernel(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float*       __restrict__ C,
                                    int M, int N)
{
    // Shared memory for the two tiles
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    // Global row & column for this thread
    const int row = blockIdx.x * TILE_SIZE + threadIdx.x;
    const int col = blockIdx.y * TILE_SIZE + threadIdx.y;

    float sum = 0.0f;

    // Number of tiles along the K‑dimension (N)
    const int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; ++t)
    {
        // ---- load tile from A (row stays constant, column varies) ----
        int aCol = t * TILE_SIZE + threadIdx.y;
        if (row < M && aCol < N)
            As[threadIdx.x][threadIdx.y] = A[row * N + aCol];
        else
            As[threadIdx.x][threadIdx.y] = 0.0f;

        // ---- load tile from B (row varies, column stays constant) ----
        int bRow = t * TILE_SIZE + threadIdx.x;
        if (bRow < N && col < M)
            Bs[threadIdx.x][threadIdx.y] = B[bRow * M + col];
        else
            Bs[threadIdx.x][threadIdx.y] = 0.0f;

        __syncthreads();

        // ---- compute partial dot product for this tile ----
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k)
            sum += As[threadIdx.x][k] * Bs[k][threadIdx.y];

        __syncthreads();
    }

    // Write result if inside the output matrix
    if (row < M && col < M)
        C[row * M + col] = sum;
}

// Host wrapper that launches the kernel
void tiled_matmul(int M, int N,
                  const torch::Tensor& A,
                  const torch::Tensor& B,
                  torch::Tensor& C)
{
    const float* A_ptr = A.data_ptr<float>();
    const float* B_ptr = B.data_ptr<float>();
    float*       C_ptr = C.data_ptr<float>();

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((M + TILE_SIZE - 1) / TILE_SIZE,
              (M + TILE_SIZE - 1) / TILE_SIZE);

    tiled_matmul_kernel<<<grid, block>>>(A_ptr, B_ptr, C_ptr, M, N);
}
"""

# -------------------------------------------------------------------------
# C++ binding – exposes the wrapper to Python
cpp_source = r"""
#include <torch/extension.h>

void tiled_matmul(int M, int N,
                  const torch::Tensor& A,
                  const torch::Tensor& B,
                  torch::Tensor& C);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tiled_matmul", &tiled_matmul,
          "Tiled matrix multiplication using shared memory");
}
"""

# -------------------------------------------------------------------------
# Build the inline CUDA extension
matmul_ext = load_inline(
    name='matmul_tiled',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# -------------------------------------------------------------------------
# The function that will be imported and evaluated
def functional_model(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Custom matrix multiplication using a shared‑memory tiled CUDA kernel.
    """
    # Ensure the inputs reside on the GPU
    if not A.is_cuda:
        A = A.cuda()
    if not B.is_cuda:
        B = B.cuda()

    M = A.size(0)   # rows of A / rows of C
    N = A.size(1)   # cols of A / rows of B

    # Allocate output matrix on the GPU
    C = torch.empty((M, M), dtype=A.dtype, device=A.device)

    # Launch the custom tiled kernel
    matmul_ext.tiled_matmul(M, N, A, B, C)

    # Ensure the kernel finishes before we return (optional, but safe)
    torch.cuda.synchronize()
    return C


# -------------------------------------------------------------------------
# Helpers for the benchmark harness (not used by functional_model)
def get_init_inputs():
    return []


def get_inputs():
    # Generate on CPU; functional_model will move them to GPU
    A = torch.rand(M, N)
    B = torch.rand(N, M)
    return [A, B]
