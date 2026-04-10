# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_090325/code_7.py
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
# Inline CUDA source – Tiled Matrix Multiplication Kernel
# -------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// Tile size optimized for TLP and shared memory occupancy on Turing (RTX 2080 Ti)
constexpr int TILE = 32;

__global__ void bmm_tiled_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float*       __restrict__ C,
    const int M,
    const int K,
    const int N)
{
    // Each block processes one M x N tile of the output (for a single batch element)
    // blockIdx.x = Batch, blockIdx.y = Output Tile Row, blockIdx.z = Output Tile Col
    const int b = blockIdx.x;
    const int ty = threadIdx.y;
    const int tx = threadIdx.x;

    const int row = blockIdx.y * TILE + ty;
    const int col = blockIdx.z * TILE + tx;

    extern __shared__ float s[];
    float* sA = s;
    float* sB = s + TILE * TILE;

    float acc = 0.0f;

    // Pointer offsets for the specific batch
    const float* A_ptr = A + b * M * K;
    const float* B_ptr = B + b * K * N;
    float* C_ptr = C + b * M * N;

    const int num_k_tiles = (K + TILE - 1) / TILE;

    for (int kt = 0; kt < num_k_tiles; ++kt) {
        // Load A tile
        if (row < M && (kt * TILE + tx) < K) {
            sA[ty * TILE + tx] = A_ptr[row * K + (kt * TILE + tx)];
        } else {
            sA[ty * TILE + tx] = 0.0f;
        }

        // Load B tile
        if (col < N && (kt * TILE + ty) < K) {
            sB[ty * TILE + tx] = B_ptr[(kt * TILE + ty) * N + col];
        } else {
            sB[ty * TILE + tx] = 0.0f;
        }

        __syncthreads();

        // Perform dot product on tile
        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            acc += sA[ty * TILE + k] * sB[k * TILE + tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C_ptr[row * N + col] = acc;
    }
}

void fused_op_forward(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& C) {
    const int batch = A.size(0);
    const int M = A.size(1);
    const int K = A.size(2);
    const int N = B.size(2);

    dim3 block(TILE, TILE);
    dim3 grid(batch, (M + TILE - 1) / TILE, (N + TILE - 1) / TILE);
    size_t shared_mem = 2 * TILE * TILE * sizeof(float);

    bmm_tiled_kernel<<<grid, block, shared_mem>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N);
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_op_forward(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& C);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Tiled BMM kernel");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def get_init_inputs():
    return []

def get_inputs():
    batch_size, m, k, n = 128, 512, 1024, 2048
    A = torch.rand(batch_size, m, k, dtype=torch.float32, device='cuda')
    B = torch.rand(batch_size, k, n, dtype=torch.float32, device='cuda')
    return [A, B]

def functional_model(A, B):
    batch, m, k = A.shape
    n = B.shape[2]
    C = torch.empty((batch, m, n), dtype=A.dtype, device=A.device)
    fused_ext.fused_op(A, B, C)
    return C
