# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_093747/code_7.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = []
FORWARD_ARG_NAMES = ['A', 'B']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = []
REQUIRED_FLAT_STATE_NAMES = []


class ModelNew(nn.Module):
    """
    Performs batched matrix multiplication (C = A * B) where A, B, and C have the same batch dimension.
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
# CUDA kernel source – tiled batched matrix‑multiply (FP32)
# Using 32x32 tiles to better saturate SM occupancy on RTX 2080Ti
# -------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define BM 32
#define BN 32
#define BK 32

__global__ void bmm_tiled_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    int batch)
{
    // Shared memory for tiling
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    int batch_idx = blockIdx.z;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Global output coordinates
    int row = blockIdx.x * BM + tx;
    int col = blockIdx.y * BN + ty;

    float sum = 0.0f;
    int numKTiles = (K + BK - 1) / BK;

    for (int k_tile = 0; k_tile < numKTiles; ++k_tile) {
        // Load tile A into shared memory
        int a_col = k_tile * BK + ty;
        if (row < M && a_col < K)
            As[tx][ty] = A[batch_idx * M * K + row * K + a_col];
        else
            As[tx][ty] = 0.0f;

        // Load tile B into shared memory
        int b_row = k_tile * BK + tx;
        if (b_row < K && col < N)
            Bs[tx][ty] = B[batch_idx * K * N + b_row * N + col];
        else
            Bs[tx][ty] = 0.0f;

        __syncthreads();

        // Compute dot product for this tile
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            sum += As[tx][k] * Bs[k][ty];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[batch_idx * M * N + row * N + col] = sum;
    }
}

void bmm_tiled_launch(
    const torch::Tensor& A,
    const torch::Tensor& B,
    torch::Tensor& C)
{
    int batch = A.size(0);
    int M = A.size(1);
    int K = A.size(2);
    int N = B.size(2);

    dim3 block(BM, BN);
    dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN, batch);

    bmm_tiled_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K, batch
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void bmm_tiled_launch(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& C);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bmm_tiled", &bmm_tiled_launch, "Tiled batched matrix multiplication");
}
"""

# Compile the extension
bmm_ext = load_inline(
    name='bmm_tiled_kernel',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(A, B):
    # Ensure inputs are on target device, contiguous, and float32
    A = A.to(device='cuda', dtype=torch.float32).contiguous()
    B = B.to(device='cuda', dtype=torch.float32).contiguous()
    
    batch, M, K = A.shape
    _, _, N = B.shape
    
    C = torch.empty((batch, M, N), device='cuda', dtype=torch.float32)
    
    # Kernel handles batched computation
    bmm_ext.bmm_tiled(A, B, C)
    
    return C
