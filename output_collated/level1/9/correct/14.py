# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_095613/code_2.py
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# --- CUDA Kernel and C++ Binding ---
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

__global__ void my_matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int M,
    const int N
) {
    // Grid-stride loop over output elements
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int idx = row; idx < M * M; idx += stride) {
        const int i = idx / M;  // row index for C
        const int j = idx % M;  // col index for C

        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[i * N + k] * B[k * M + j];
        }
        C[idx] = sum;
    }
}

void my_matmul_forward(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C
) {
    const int M = A.size(0);
    const int N = A.size(1);

    const int threads_per_block = 256;
    const int blocks = (M * M + threads_per_block - 1) / threads_per_block;

    my_matmul_kernel<<<blocks, threads_per_block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M,
        N
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
    // Note: Synchronization is typically done at a higher level if needed for timing
}
"""

cpp_source = r"""
#include <torch/extension.h>

void my_matmul_forward(torch::Tensor A, torch::Tensor B, torch::Tensor C);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("my_matmul", &my_matmul_forward, "Matrix multiply kernel (MxN) * (NxM) -> (MxM)");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# --- Model Parameters ---
M = 16384 * 2
N = 16 * 2

# --- Input Generation (Unchanged) ---
def get_init_inputs():
    return []  # No special initialization inputs needed

def get_inputs():
    A = torch.rand(M, N, dtype=torch.float32, device='cuda')
    B = torch.rand(N, M, dtype=torch.float32, device='cuda')
    return [A, B]

# --- Optimized Functional Model ---
def functional_model(A, B):
    """
    Performs matrix multiplication A @ B using a custom CUDA kernel.
    A: (M, N)
    B: (N, M)
    Returns: C (M, M)
    """
    # Ensure inputs are on CUDA and float32 for our kernel
    assert A.is_cuda and B.is_cuda, "Inputs must be on CUDA"
    assert A.dtype == torch.float32 and B.dtype == torch.float32, "Inputs must be float32"

    C = torch.empty(A.size(0), B.size(1), dtype=A.dtype, device=A.device)
    fused_ext.my_matmul(A, B, C)
    return C

