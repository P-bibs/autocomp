# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_101900/code_11.py
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
# CUDA kernel with shared-memory caching of the row scale factors
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// Kernel: row-wise broadcast multiply with A cached in shared memory
__global__ void fused_multiply_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ Out,
    int N, int M)
{
    // blockDim.y == 16, therefore we allocate 16 floats per block
    __shared__ float A_shared[16];

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Load A[row] once per block (only thread 0 of each row does it)
    if (threadIdx.x == 0 && row < N) {
        A_shared[threadIdx.y] = A[row];
    }
    __syncthreads();

    // Compute output using the cached value
    if (row < N && col < M) {
        float a = A_shared[threadIdx.y];
        Out[row * M + col] = a * B[row * M + col];
    }
}

// Host wrapper that launches the kernel
void fused_op_forward(torch::Tensor A, torch::Tensor B, torch::Tensor Out) {
    int N = A.size(0);
    int M = B.size(1);

    // 32 columns (x) and 16 rows (y) per block gives good warp occupancy
    dim3 threadsPerBlock(32, 16);
    dim3 numBlocks((M + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    fused_multiply_kernel<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), Out.data_ptr<float>(), N, M);
}
"""

# ----------------------------------------------------------------------
# C++ binding (pybind11)
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(torch::Tensor A, torch::Tensor B, torch::Tensor Out);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused broadcast multiply with shared-memory optimization");
}
"""

# ----------------------------------------------------------------------
# Compile the extension inline
# ----------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_mul_opt',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True,
)

# ----------------------------------------------------------------------
# Functional wrapper expected by the evaluation harness
# ----------------------------------------------------------------------
def functional_model(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs  Out[i, j] = A[i] * B[i, j]  for all i, j.
    A has shape (N,), B has shape (N, M), Out will have shape (N, M).
    """
    N = A.shape[0]
    M = B.shape[1]
    # Allocate output on the same device as inputs (cuda)
    out = torch.empty((N, M), device=A.device, dtype=A.dtype)
    fused_ext.fused_op(A, B, out)
    return out


# ----------------------------------------------------------------------
# Dummy input generators (required by the harness)
# ----------------------------------------------------------------------
M, N = 4096, 4096

def get_init_inputs():
    """No persistent state is required."""
    return []

def get_inputs():
    """Return a fresh pair of random matrices on GPU."""
    A = torch.rand(N, device='cuda')          # shape (N,)
    B = torch.rand(N, M, device='cuda')       # shape (N, M)
    return [A, B]

# ----------------------------------------------------------------------
# The entry point that will be exercised by the evaluation script
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple sanity-check (can be removed when running the real benchmark)
    A, B = get_inputs()
    out = functional_model(A, B)
    expected = A.view(-1, 1) * B
    assert torch.allclose(out, expected, atol=1e-5), "Result mismatch!"
    print("Correctness OK.")
