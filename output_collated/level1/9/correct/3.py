# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_094501/code_6.py
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

# ----------------------------------------------------------------------
# CUDA kernel – tiled matrix multiplication (Optimization #8)
# We tile the computation into 32x32 blocks to effectively utilize 
# shared memory and reduce global memory bandwidth pressure.
# ----------------------------------------------------------------------
TILE = 32

cuda_kernel = f"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE {TILE}

__global__ void matmul_tiled_kernel(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ C,
                                    int M, int K, int N) {{
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.0f;

    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int numTiles = (K + TILE - 1) / TILE;
    for (int t = 0; t < numTiles; ++t) {{
        // Load tile of A
        int a_col = t * TILE + threadIdx.x;
        if (row < M && a_col < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + a_col];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        // Load tile of B
        int b_row = t * TILE + threadIdx.y;
        if (b_row < K && col < N)
            Bs[threadIdx.y][threadIdx.x] = B[b_row * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE; ++i) {{
            acc += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }}

        __syncthreads();
    }}

    if (row < M && col < N) {{
        C[row * N + col] = acc;
    }}
}}

void tiled_matmul_forward(const torch::Tensor &A,
                          const torch::Tensor &B,
                          torch::Tensor &C) {{
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    dim3 threads(TILE, TILE);
    dim3 blocks((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    matmul_tiled_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, K, N
    );
}}
"""

# ----------------------------------------------------------------------
# C++ binding
# ----------------------------------------------------------------------
cpp_source = """
#include <torch/extension.h>

void tiled_matmul_forward(const torch::Tensor &A,
                          const torch::Tensor &B,
                          torch::Tensor &C);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tiled_matmul", &tiled_matmul_forward, "Tiled matrix multiplication (CUDA)");
}
"""

# Compile
fused_ext = load_inline(
    name="tiled_matmul_ext",
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True,
    verbose=False,
)

# ----------------------------------------------------------------------
# Constants and functional_model
# ----------------------------------------------------------------------
M = 16384 * 2
N = 16 * 2

def get_init_inputs():
    return []

def get_inputs():
    # K must be defined as per the usage context (the inner dimension)
    A = torch.rand(M, N, device="cuda", dtype=torch.float32)
    B = torch.rand(N, M, device="cuda", dtype=torch.float32)
    return [A, B]

def functional_model(A, B):
    # C = A @ B. Shapes: A(M, N), B(N, M) -> C(M, M)
    C = torch.empty(A.size(0), B.size(1), device=A.device, dtype=A.dtype)
    fused_ext.tiled_matmul(A, B, C)
    return C
