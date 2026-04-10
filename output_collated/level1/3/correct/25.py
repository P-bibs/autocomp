# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_092316/code_3.py
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# ----------------------------------------------------------------------
# Shapes (kept for reference – not used by the kernel)
batch_size = 128
m = 128 * 4   # 512
k = 256 * 4   # 1024
n = 512 * 4   # 2048

# ----------------------------------------------------------------------
# Inline CUDA kernel + host code
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

constexpr int TILE_SIZE = 16;

__global__ void bmm_kernel(const float* __restrict__ A,
                           const float* __restrict__ B,
                           float* __restrict__ C,
                           const int64_t batch,
                           const int64_t M,
                           const int64_t K,
                           const int64_t N)
{
    // block indices
    const int64_t b = blockIdx.z;          // batch index
    const int64_t rowBlock = blockIdx.y;   // output row tile
    const int64_t colBlock = blockIdx.x;   // output column tile

    // thread indices inside the block
    const int row = rowBlock * blockDim.y + threadIdx.y;
    const int col = colBlock * blockDim.x + threadIdx.x;

    // shared memory tiles with padding to avoid bank conflicts
    __shared__ float tileA[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE + 1];

    float sum = 0.0f;

    // loop over the K dimension in tiles
    for (int64_t kTile = 0; kTile < K; kTile += TILE_SIZE) {
        // ---- load tile from A -------------------------------------------------
        if (row < M && (kTile + threadIdx.x) < K) {
            tileA[threadIdx.y][threadIdx.x] =
                A[b * M * K + row * K + (kTile + threadIdx.x)];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // ---- load tile from B -------------------------------------------------
        if ((kTile + threadIdx.y) < K && col < N) {
            tileB[threadIdx.y][threadIdx.x] =
                B[b * K * N + (kTile + threadIdx.y) * N + col];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // ---- compute partial dot product --------------------------------------
        for (int i = 0; i < TILE_SIZE; ++i) {
            sum += tileA[threadIdx.y][i] * tileB[i][threadIdx.x];
        }

        __syncthreads();
    }

    // ---- write result --------------------------------------------------------
    if (row < M && col < N) {
        C[b * M * N + row * N + col] = sum;
    }
}

// Host function that launches the kernel
at::Tensor bmm_forward(const at::Tensor& A_, const at::Tensor& B_) {
    auto A = A_.contiguous();
    auto B = B_.contiguous();

    const int64_t batch = A.size(0);
    const int64_t M     = A.size(1);
    const int64_t K     = A.size(2);
    const int64_t N     = B.size(2);

    // allocate output
    auto C = at::empty({batch, M, N}, A.options());

    const int blockSize = TILE_SIZE;               // 16
    dim3 block(blockSize, blockSize);
    dim3 grid((N + blockSize - 1) / blockSize,
              (M + blockSize - 1) / blockSize,
              batch);

    bmm_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        batch, M, K, N);

    // Removed cudaDeviceSynchronize() for better performance
    return C;
}
"""

# ----------------------------------------------------------------------
# C++ binding – exposes the host function to Python
cpp_source = r"""
#include <torch/extension.h>

at::Tensor bmm_forward(const at::Tensor& A, const at::Tensor& B);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bmm", &bmm_forward, "Batch matrix multiplication using a custom CUDA kernel");
}
"""

# ----------------------------------------------------------------------
# Build the inline extension
fused_ext = load_inline(
    name='fused_bmm',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# ----------------------------------------------------------------------
# The function that will be imported and evaluated
def functional_model(A, B):
    """
    Custom batch matrix multiplication: A (batch, m, k) × B (batch, k, n) -> C (batch, m, n)
    Uses a hand-written CUDA kernel instead of torch.bmm.
    """
    # Ensure the inputs are on the GPU and are contiguous
    if not A.is_cuda:
        A = A.cuda()
    if not B.is_cuda:
        B = B.cuda()
    A = A.contiguous()
    B = B.contiguous()

    # Call the compiled CUDA kernel
    C = fused_ext.bmm(A, B)
    return C


# ----------------------------------------------------------------------
# Helper functions required by the test harness (not used by functional_model)
def get_init_inputs():
    return []


def get_inputs():
    A = torch.rand(batch_size, m, k)
    B = torch.rand(batch_size, k, n)
    return [A, B]
