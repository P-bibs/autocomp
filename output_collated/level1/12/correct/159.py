# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_110950/code_31.py
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
# Optimized for coalesced memory access and register reuse.
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
    // Flattening the 2D grid to ensure each row-segment is processed 
    // by a warp or block in a contiguous fashion.
    // Each thread calculates one element.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * M;

    if (idx < total_elements) {
        int row = idx / M;
        int col = idx % M;
        // Broadcast A[row] effectively acts as a constant for the row
        C[idx] = A[row] * B[idx];
    }
}

void scale_rows_cuda(const torch::Tensor& A,
                     const torch::Tensor& B,
                     torch::Tensor& C)
{
    const int N = A.size(0);
    const int M = B.size(1);
    const int total_elements = N * M;

    // Use 256 threads per block, which is standard for occupancy
    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;

    scale_rows_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N, M);
}
"""

# ----------------------------------------------------------------------
# C++ binding (PYBIND11)
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
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# ----------------------------------------------------------------------
# Optimized functional model 
# ----------------------------------------------------------------------
def functional_model(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Optimized implementation of A.unsqueeze(1) * B using a custom CUDA kernel.
    Ensures memory-coalesced access leading to peak bandwidth utilization.
    """
    # Allocate output tensor on the same device as B
    C = torch.empty_like(B)
    
    # Execute the kernel
    scale_rows_ext.scale_rows(A, B, C)
    
    return C

# ----------------------------------------------------------------------
# Boilerplate requirements (N, M, get_inputs, get_init_inputs)
# ----------------------------------------------------------------------
M = 4096
N = 4096

def get_init_inputs():
    return []

def get_inputs():
    A = torch.rand(N).cuda()
    B = torch.rand(N, M).cuda()
    return [A, B]
