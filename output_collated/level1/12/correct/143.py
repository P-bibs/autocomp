# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_110950/code_15.py
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
# CUDA kernel – scales each row of B by the corresponding element of A
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void scale_rows_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N, int M)
{
    // blockIdx.x -> row index
    // blockIdx.y -> column block index
    int row = blockIdx.x;
    int col = blockIdx.y * blockDim.x + threadIdx.x;

    if (row < N && col < M) {
        C[row * M + col] = A[row] * B[row * M + col];
    }
}

void scale_rows_cuda(const torch::Tensor& A,
                     const torch::Tensor& B,
                     torch::Tensor& C)
{
    const int N = A.size(0);
    const int M = B.size(1);

    const int BLOCK = 256;                     // threads per block
    dim3 block(BLOCK);
    dim3 grid(N, (M + BLOCK - 1) / BLOCK);     // 2‑D grid: (rows, column‑blocks)

    scale_rows_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N, M);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
    }
}
"""

# ----------------------------------------------------------------------
# C++ binding (PYBIND11) – exposes the kernel to Python
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void scale_rows_cuda(const torch::Tensor& A,
                     const torch::Tensor& B,
                     torch::Tensor& C);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("scale_rows", &scale_rows_cuda, "Scale rows of B by A (CUDA)");
}
"""

# ----------------------------------------------------------------------
# Compile the inline CUDA extension
# ----------------------------------------------------------------------
scale_rows_ext = load_inline(
    name='scale_rows',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],   # required compilation flags
    with_cuda=True
)

# ----------------------------------------------------------------------
# Functional model – replaces the original PyTorch broadcast
# ----------------------------------------------------------------------
def functional_model(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Returns  A.unsqueeze(1) * B   using a custom coalesced CUDA kernel.
    """
    # Ensure inputs are on the GPU
    if not A.is_cuda:
        A = A.cuda()
    if not B.is_cuda:
        B = B.cuda()

    # Allocate output on the same device
    C = torch.empty_like(B)   # shape (N, M)

    # Launch the fused kernel
    scale_rows_ext.scale_rows(A, B, C)

    return C


# ----------------------------------------------------------------------
# Boilerplate – matches the original interface
# ----------------------------------------------------------------------
M = 4096
N = 4096

def get_init_inputs():
    return []                     # no special initialisation

def get_inputs():
    A = torch.rand(N)            # 1‑D vector of length N
    B = torch.rand(N, M)         # 2‑D matrix N×M
    return [A, B]
