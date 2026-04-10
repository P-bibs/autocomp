# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_101225/code_7.py
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

# Problem dimensions
M = 16384 * 2   # 32768
N = 16 * 2      # 32

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_M 16
#define TILE_N 32

__global__ void gemm_kernel(const float* __restrict__ A,
                            const float* __restrict__ B,
                            float*       __restrict__ C,
                            int M, int N) {
    // Shared memory for tiles
    __shared__ float As[TILE_M][TILE_N];
    __shared__ float Bs[TILE_N][TILE_M];

    int tx = threadIdx.x; // maps to columns in C
    int ty = threadIdx.y; // maps to rows in C
    int row = blockIdx.y * TILE_M + ty;
    int col = blockIdx.x * TILE_M + tx;

    float acc = 0.0f;

    // We process K=N in one go since N is small (32)
    // Load A tile (M x N) and B tile (N x M)
    // A is M x N, so A tile is loaded row-wise
    if (row < M && tx < N) As[ty][tx] = A[row * N + tx];
    else As[ty][tx] = 0.0f;

    // B is N x M, so B tile is loaded row-wise
    if (ty < N && col < M) Bs[ty][tx] = B[ty * M + col];
    else Bs[ty][tx] = 0.0f;

    __syncthreads();

    // Compute dot product along the N dimension
    #pragma unroll
    for (int k = 0; k < N; ++k) {
        acc += As[ty][k] * Bs[k][tx];
    }

    if (row < M && col < M) {
        C[row * M + col] = acc;
    }
}

void gemm_forward(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& C) {
    const int M = A.size(0);
    const int N = A.size(1);
    
    dim3 block(16, 16);
    dim3 grid((M + 15) / 16, (M + 15) / 16);
    
    gemm_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void gemm_forward(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& C);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm", &gemm_forward, "Custom Tiled GEMM");
}
"""

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
    A = torch.rand(M, N, dtype=torch.float32)
    B = torch.rand(N, M, dtype=torch.float32)
    return [A, B]

def functional_model(A, B):
    # Ensure inputs are on GPU
    if A.device.type != 'cuda': A = A.cuda()
    if B.device.type != 'cuda': B = B.cuda()
    
    C = torch.empty((M, M), dtype=torch.float32, device='cuda')
    fused_ext.gemm(A, B, C)
    return C
