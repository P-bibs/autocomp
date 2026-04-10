# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_102701/code_15.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = []
FORWARD_ARG_NAMES = ['A', 'B']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = []
REQUIRED_FLAT_STATE_NAMES = []


class ModelNew(nn.Module):
    """
    Simple model that performs a matrix multiplication of a diagonal matrix with another matrix.
    C = diag(A) * B
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
# CUDA kernel – tiled with shared-memory caching of A
# ----------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_unsqueeze_multiply_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N,
    int M
) {
    // Shared memory for caching A values belonging to the rows of this block.
    extern __shared__ float s_A[];

    // Global row / column for this thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Out-of-bounds guard
    if (row >= N || col >= M) return;

    // ---- Load A[row] into shared memory (once per row in the block) ----
    if (threadIdx.x == 0) {
        s_A[threadIdx.y] = A[row];
    }
    __syncthreads();

    // ---- Retrieve the cached A value for our row ----
    float a_val = s_A[threadIdx.y];

    // ---- Load B element using the read-only data cache ----
    float b_val = __ldg(&B[row * M + col]);

    // ---- Compute and write the result ----
    C[row * M + col] = a_val * b_val;
}

// Host-side launcher
void fused_unsqueeze_multiply_forward(
    const torch::Tensor& A,
    const torch::Tensor& B,
    torch::Tensor& C,
    int N,
    int M
) {
    // 16 × 16 thread block (256 threads)
    const int BLOCK_X = 16;
    const int BLOCK_Y = 16;
    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid((M + BLOCK_X - 1) / BLOCK_X,
              (N + BLOCK_Y - 1) / BLOCK_Y);

    // Shared memory: one float per row in the block
    int shared_mem = BLOCK_Y * sizeof(float);

    fused_unsqueeze_multiply_kernel<<<grid, block, shared_mem>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N,
        M
    );

    // Kernel-launch error check
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("CUDA kernel launch failed: ", cudaGetErrorString(err));
    }
}
"""

# ----------------------------------------------------------------------
# C++ binding (PyBind11)
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void fused_unsqueeze_multiply_forward(
    const torch::Tensor& A,
    const torch::Tensor& B,
    torch::Tensor& C,
    int N,
    int M
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_unsqueeze_multiply",
          &fused_unsqueeze_multiply_forward,
          "Fused unsqueeze-multiply (optimized with shared-memory A cache)");
}
"""

# ----------------------------------------------------------------------
# Compile the inline extension
# ----------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_unsqueeze_multiply',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# ----------------------------------------------------------------------
# Functional wrapper that will be called during evaluation
# ----------------------------------------------------------------------
def functional_model(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Return C = A_unsqueezed * B, using the optimized CUDA kernel."""
    N, M = B.shape                     # N = rows, M = columns
    C = torch.empty(N, M, dtype=torch.float32, device='cuda')

    # Ensure inputs are float32 and on the GPU
    A_gpu = A.to(torch.float32).cuda()
    B_gpu = B.to(torch.float32).cuda()

    # Launch the optimized kernel
    fused_ext.fused_unsqueeze_multiply(A_gpu, B_gpu, C, N, M)

    return C

# ----------------------------------------------------------------------
# Helper functions required by the benchmark harness
# ----------------------------------------------------------------------
M = 4096
N = 4096

def get_init_inputs():
    return []          # No special initialization needed

def get_inputs():
    A = torch.rand(N)          # 1-D vector of length N
    B = torch.rand(N, M)       # N×M matrix
    return [A, B]
