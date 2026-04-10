# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_091341/code_6.py
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

# Optimization: Grid-stride loop batched GEMM
# Tile size: 64x64, Thread block: 64x64 (internal blocking logic)
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TM 64
#define TN 64
#define BK 32

__global__ void batched_gemm_forward_kernel(const float* __restrict__ A,
                                             const float* __restrict__ B,
                                             float* __restrict__ C,
                                             int batch_size,
                                             int M, int K, int N) {
    // Shared memory tiles
    __shared__ float As[TM][BK];
    __shared__ float Bs[BK][TN];

    const int ty = threadIdx.y;
    const int tx = threadIdx.x;

    // Grid-stride over the batch dimension
    for (int batch = blockIdx.z; batch < batch_size; batch += gridDim.z) {
        float acc = 0.0f;
        
        // Tile output coordinates
        const int row_base = blockIdx.y * TM;
        const int col_base = blockIdx.x * TN;

        // Iterate over K dimension
        for (int t = 0; t < K; t += BK) {
            // Load A tile
            if (row_base + ty < M && t + tx < K)
                As[ty][tx] = A[batch * M * K + (row_base + ty) * K + (t + tx)];
            else
                As[ty][tx] = 0.0f;

            // Load B tile
            if (t + ty < K && col_base + tx < N)
                Bs[ty][tx] = B[batch * K * N + (t + ty) * N + (col_base + tx)];
            else
                Bs[ty][tx] = 0.0f;

            __syncthreads();

            #pragma unroll
            for (int p = 0; p < BK; ++p) {
                acc += As[ty][p] * Bs[p][tx];
            }
            __syncthreads();
        }

        // Store result
        if (row_base + ty < M && col_base + tx < N) {
            C[batch * M * N + (row_base + ty) * N + (col_base + tx)] = acc;
        }
    }
}

torch::Tensor batched_gemm(torch::Tensor A, torch::Tensor B) {
    int batch_size = A.size(0);
    int M = A.size(1);
    int K = A.size(2);
    int N = B.size(2);

    auto options = A.options();
    auto C = torch::zeros({batch_size, M, N}, options);

    dim3 threads(TN, TM); // 64, 64
    dim3 blocks((N + TN - 1) / TN, (M + TM - 1) / TM, 4); // 4 chunks in Z index

    batched_gemm_forward_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
        batch_size, M, K, N
    );
    
    return C;
}
"""

cpp_source = r"""
#include <torch/extension.h>
torch::Tensor batched_gemm(torch::Tensor A, torch::Tensor B);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("batched_gemm", &batched_gemm, "Batched GEMM with grid-stride loop");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_gemm_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

batch_size = 128
m = 128 * 4
k = 256 * 4
n = 512 * 4

def get_init_inputs():
    return []

def get_inputs():
    torch.manual_seed(42)
    A = torch.rand(batch_size, m, k, device='cuda')
    B = torch.rand(batch_size, k, n, device='cuda')
    return [A, B]

def functional_model(A, B):
    return fused_ext.batched_gemm(A, B)
