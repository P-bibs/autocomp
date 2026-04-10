# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_104927/code_15.py
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

# ------------------------------------------------------------
# 1. CUDA kernel (inline string)
# ------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// Grid-stride kernel that computes  A[row] * B[row, col]  for all (row, col)
__global__ void mul_kernel(const float* __restrict__ A,
                           const float* __restrict__ B,
                           float*       __restrict__ C,
                           const int N,
                           const int M)
{
    // Global linear index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * M;
    const int stride = blockDim.x * gridDim.x;   // total number of threads

    for (int i = idx; i < total; i += stride) {
        int row = i / M;
        int col = i % M;
        float a = A[row];                     // broadcast within warp
        float b = B[row * M + col];           // coalesced read of B
        C[i] = a * b;
    }
}

// Host wrapper that sets up the launch configuration
void mul_cuda(const torch::Tensor& A,
              const torch::Tensor& B,
              torch::Tensor& C)
{
    const int N = A.size(0);
    const int M = B.size(1);
    const int total = N * M;

    const float* A_ptr = A.data_ptr<float>();
    const float* B_ptr = B.data_ptr<float>();
    float*       C_ptr = C.data_ptr<float>();

    // Choose a block size that is a power-of-2 and gives good occupancy
    const int block_size = 1024;                     // 1024 threads per block
    // Compute required number of blocks, cap at CUDA's limit (65535)
    int grid_size = (total + block_size - 1) / block_size;
    if (grid_size > 65535) grid_size = 65535;

    mul_kernel<<<grid_size, block_size>>>(A_ptr, B_ptr, C_ptr, N, M);
}
"""

# ------------------------------------------------------------
# 2. C++ bindings (PyBind11)
# ------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void mul_cuda(const torch::Tensor& A,
              const torch::Tensor& B,
              torch::Tensor& C);

// Python-visible function that dispatches to the CUDA kernel
void mul(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    mul_cuda(A, B, C);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mul", &mul, "Fused A[row] * B[row, col] kernel");
}
"""

# ------------------------------------------------------------
# 3. Compile the inline extension
# ------------------------------------------------------------
fused_ext = load_inline(
    name='fused_mul',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# ------------------------------------------------------------
# 4. Functional model that uses the custom CUDA kernel
# ------------------------------------------------------------
def functional_model(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Returns A.unsqueeze(1) * B  (shape: N x M)
    Implemented with a custom CUDA kernel for maximal memory bandwidth.
    """
    # Move inputs to GPU if they are not already there
    if not A.is_cuda:
        A = A.cuda()
    if not B.is_cuda:
        B = B.cuda()

    # Ensure B is contiguous (row-major) so the raw pointer is valid
    B = B.contiguous()

    N = A.shape[0]          # vector length
    M = B.shape[1]          # number of columns

    # Allocate output on the same device
    C = torch.empty((N, M), dtype=A.dtype, device=A.device)

    # Launch the fused CUDA kernel
    fused_ext.mul(A, B, C)

    return C


# ------------------------------------------------------------
# 5. Helper functions required by the benchmark harness
# ------------------------------------------------------------
M = 4096
N = 4096

def get_init_inputs():
    """No special initialization inputs are needed."""
    return []

def get_inputs():
    """Return the inputs used in the original benchmark."""
    A = torch.rand(N)          # 1-D vector
    B = torch.rand(N, M)       # matrix
    return [A, B]
