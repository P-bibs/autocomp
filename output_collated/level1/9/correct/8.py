# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_094844/code_7.py
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
# Constants for the problem
# M = 32768, N = 32. 
# Result is M x M (32768 x 32768)
# Optimization: Tiled GEMM in CUDA.
# ----------------------------------------------------------------------

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_DIM 16
#define K_DIM 32

__global__ void gemm_kernel(const float* __restrict__ A,
                            const float* __restrict__ B,
                            float*       __restrict__ C,
                            int M, int K)
{
    // Shared memory: 
    // As is [TILE_DIM][K_DIM] (16*32*4 bytes = 2KB)
    // Bs is [K_DIM][TILE_DIM] (32*16*4 bytes = 2KB)
    __shared__ float As[TILE_DIM][K_DIM];
    __shared__ float Bs[K_DIM][TILE_DIM];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_DIM + ty;
    int col = blockIdx.x * TILE_DIM + tx;

    float sum = 0.0f;

    // A is M x K, B is K x M
    // Since K is small (32), we iterate through columns of A and rows of B
    // Load tile of A: (TILE_DIM x K)
    for (int k = 0; k < K; k += TILE_DIM) {
        if (row < M && (k + tx) < K) {
            As[ty][tx] = A[row * K + (k + tx)];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        // Load tile of B: (K x TILE_DIM)
        if ((k + ty) < K && col < M) {
            Bs[ty][tx] = B[(k + ty) * M + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_DIM; ++i) {
            sum += As[ty][i] * Bs[i][tx];
        }
        __syncthreads();
    }

    if (row < M && col < M) {
        C[row * M + col] = sum;
    }
}

torch::Tensor fused_op_forward(const torch::Tensor& A, const torch::Tensor& B) {
    const int M = A.size(0);
    const int K = A.size(1);
    
    auto output = torch::empty({M, M}, A.options());
    
    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((M + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);
    
    gemm_kernel<<<grid, block>>>(
        A.data_ptr<float>(), 
        B.data_ptr<float>(), 
        output.data_ptr<float>(), 
        M, K
    );
    
    return output;
}
"""

cpp_source = r"""
#include <torch/extension.h>
torch::Tensor fused_op_forward(const torch::Tensor& A, const torch::Tensor& B);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Tiled GEMM Kernel");
}
"""

fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(A, B):
    """
    Computes C = A @ B using custom tiled CUDA kernel.
    """
    # Ensure inputs are on GPU and contiguous
    A_gpu = A.cuda()
    B_gpu = B.cuda()
    return fused_ext.fused_op(A_gpu, B_gpu)
