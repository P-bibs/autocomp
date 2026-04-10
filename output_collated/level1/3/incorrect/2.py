# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_092316/code_5.py
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

# The implementation focuses on Shared Memory tiling.
# We partition the workload such that each thread block computes a 64x64 output tile.
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_M 64
#define TILE_N 64
#define TILE_K 16

__global__ void bmm_kernel(const float* A, const float* B, float* C, int M, int K, int N) {
    int batch = blockIdx.z;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    __shared__ float sA[TILE_M][TILE_K];
    __shared__ float sB[TILE_K][TILE_N];

    float acc[4][4] = {0.0f}; 
    const float* A_ptr = A + batch * M * K;
    const float* B_ptr = B + batch * K * N;
    float* C_ptr = C + batch * M * N;

    int col_offset = blockIdx.x * TILE_N;
    int row_offset = blockIdx.y * TILE_M;

    for (int k_step = 0; k_step < K; k_step += TILE_K) {
        // Load tiles into shared memory
        #pragma unroll
        for (int i = 0; i < TILE_M; i += blockDim.y)
            sA[ty + i][tx] = A_ptr[(row_offset + ty + i) * K + k_step + tx];
        
        #pragma unroll
        for (int i = 0; i < TILE_K; i += blockDim.y)
            sB[ty + i][tx] = B_ptr[(k_step + ty + i) * N + col_offset + tx];

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_K; ++k) {
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j)
                    acc[i][j] += sA[ty + i * 16][k] * sB[k][tx + j * 16];
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            C_ptr[(row_offset + ty + i * 16) * N + (col_offset + tx + j * 16)] = acc[i][j];
}

void bmm_launcher(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    int batch = A.size(0);
    int M = A.size(1);
    int K = A.size(2);
    int N = B.size(2);
    
    // Grid dimensions: 64x64 tiles
    dim3 threads(16, 16);
    dim3 blocks(N / TILE_N, M / TILE_M, batch);
    
    bmm_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void bmm_launcher(torch::Tensor A, torch::Tensor B, torch::Tensor C);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bmm_launcher", &bmm_launcher, "Tiled BMM kernel");
}
"""

# Compile the extension
bmm_ext = load_inline(
    name='bmm_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(A, B):
    # Ensure inputs are contiguous for row-major global memory access
    A_contig = A.contiguous()
    B_contig = B.contiguous()
    C = torch.zeros((A.shape[0], A.shape[1], B.shape[2]), device=A.device, dtype=A.dtype)
    
    bmm_ext.bmm_launcher(A_contig, B_contig, C)
    return C

batch_size = 128
m = 128 * 4
k = 256 * 4
n = 512 * 4

def get_init_inputs():
    return []

def get_inputs():
    A = torch.rand(batch_size, m, k, device='cuda', dtype=torch.float32)
    B = torch.rand(batch_size, k, n, device='cuda', dtype=torch.float32)
    return [A, B]
