# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_093329/code_3.py
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

# ----------------------------------------------------------------------
# Problem sizes (same as the original script)
# ----------------------------------------------------------------------
batch_size = 128
m = 128 * 4   # 512
k = 256 * 4   # 1024
n = 512 * 4   # 2048

# ----------------------------------------------------------------------
# CUDA source – tiled batched GEMM with coalesced memory access
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

__global__ void bmm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N,
    int strideA, int strideB, int strideC)
{
    // batch index
    const int batch = blockIdx.z;

    // row/col of the output tile this block is responsible for
    const int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    const int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    // shared memory for the current tiles of A and B
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    float sum = 0.0f;

    // number of tiles along the K dimension
    const int tilesK = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int t = 0; t < tilesK; ++t) {
        // ----- coalesced load of A tile -----
        int aCol = t * BLOCK_SIZE + threadIdx.x;
        if (row < M && aCol < K) {
            // __ldg leverages the read-only data cache
            As[threadIdx.y][threadIdx.x] = __ldg(&A[batch * strideA + row * K + aCol]);
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // ----- coalesced load of B tile -----
        int bRow = t * BLOCK_SIZE + threadIdx.y;
        if (bRow < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = __ldg(&B[batch * strideB + bRow * N + col]);
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // ----- compute dot product for this tile -----
        #pragma unroll
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }

        __syncthreads();
    }

    // ----- write back the result -----
    if (row < M && col < N) {
        C[batch * strideC + row * N + col] = sum;
    }
}

// Wrapper that can be called from Python
void bmm_cuda(const at::Tensor A_, const at::Tensor B_, at::Tensor C_) {
    // Ensure tensors are contiguous (row-major) and reside on GPU
    auto A = A_.contiguous();
    auto B = B_.contiguous();
    auto C = C_.contiguous();

    const float* A_ptr = A.data_ptr<float>();
    const float* B_ptr = B.data_ptr<float>();
    float*       C_ptr = C.data_ptr<float>();

    const int M   = A.size(1);
    const int K   = A.size(2);
    const int N   = B.size(2);
    const int batch = A.size(0);

    const int strideA = M * K;   // batch stride for A
    const int strideB = K * N;   // batch stride for B
    const int strideC = M * N;   // batch stride for C

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid( (N + BLOCK_SIZE - 1) / BLOCK_SIZE,
               (M + BLOCK_SIZE - 1) / BLOCK_SIZE,
               batch );

    bmm_kernel<<<grid, block>>>(A_ptr, B_ptr, C_ptr,
                                M, K, N,
                                strideA, strideB, strideC);
    cudaDeviceSynchronize();
}
"""

# ----------------------------------------------------------------------
# C++ binding – expose the kernel to Python via PyBind11
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void bmm_cuda(const at::Tensor A, const at::Tensor B, at::Tensor C);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bmm_cuda", &bmm_cuda,
          "Custom batched matrix multiplication (coalesced GPU kernel)");
}
"""

# ----------------------------------------------------------------------
# Compile the inline CUDA extension
# ----------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# ----------------------------------------------------------------------
# The functional_model that will be evaluated
# ----------------------------------------------------------------------
def functional_model(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs the batch matrix multiplication C = A @ B using a custom
    coalesced CUDA kernel.
    """
    # Ensure tensors are on the GPU and are float32
    if A.device.type != 'cuda':
        A = A.cuda()
    if B.device.type != 'cuda':
        B = B.cuda()

    A = A.float()
    B = B.float()

    batch, M, K = A.shape          # A: (batch, M, K)
    _,        _, N = B.shape      # B: (batch, K, N)

    # Allocate output tensor
    C = torch.empty(batch, M, N, dtype=torch.float32, device=A.device)

    # Call the compiled CUDA kernel
    fused_ext.bmm_cuda(A, B, C)

    return C

# ----------------------------------------------------------------------
# Helpers required by the harness
# ----------------------------------------------------------------------
def get_init_inputs():
    """No special initialization inputs are needed."""
    return []

def get_inputs():
    """Generate random input tensors on the GPU."""
    A = torch.rand(batch_size, m, k, dtype=torch.float32).cuda()
    B = torch.rand(batch_size, k, n, dtype=torch.float32).cuda()
    return [A, B]
