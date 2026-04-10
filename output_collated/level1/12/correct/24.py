# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_101900/code_21.py
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
# CUDA kernel and host code
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// Kernel: each block handles one row i of the matrix B.
// A[i] is loaded once per block into register/shared memory 
// to avoid redundant global memory reads.
__global__ void scale_row_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float*       __restrict__ C,
    int N,
    int M)
{
    int i = blockIdx.x;
    if (i >= N) return;

    // Cache A[i] in a register; this is effectively shared memory
    // scoped to the block's lifespan for this row index.
    float a_val = A[i];

    // Each thread processes elements in the row strided by blockDim.x.
    // This provides coalesced memory access to B and C.
    int row_offset = i * M;
    for (int j = threadIdx.x; j < M; j += blockDim.x) {
        C[row_offset + j] = a_val * B[row_offset + j];
    }
}

void fused_scale_forward(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    const int N = A.size(0);
    const int M = B.size(1);
    
    // 256 threads per block is a sweet spot for RTX 2080 Ti occupancy
    const int threads = 256;
    const int blocks = N;

    scale_row_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N,
        M
    );
}
"""

# ----------------------------------------------------------------------
# C++ binding (pybind11)
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void fused_scale_forward(torch::Tensor A, torch::Tensor B, torch::Tensor C);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_scale", &fused_scale_forward, "Fused row-wise scaling kernel");
}
"""

# Compile the inline extension
fused_ext = load_inline(
    name='fused_scale',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Computes A.unsqueeze(1) * B using a custom CUDA kernel.
    Optimized to eliminate redundant reads of A using block-indexed kernel.
    """
    # Ensure inputs are on the same GPU device
    if not A.is_cuda:
        A = A.cuda()
    if not B.is_cuda:
        B = B.cuda()

    # Pre-allocate output to avoid internal PyTorch reallocations
    C = torch.empty_like(B)

    # Launch the fused kernel via the C++ binding
    fused_ext.fused_scale(A, B, C)
    
    return C

def get_init_inputs():
    return []

def get_inputs():
    N, M = 4096, 4096
    A = torch.rand(N, device='cuda')
    B = torch.rand(N, M, device='cuda')
    return [A, B]
