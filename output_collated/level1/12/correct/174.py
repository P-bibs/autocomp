# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_112116/code_15.py
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

# -------------------------------------------------------------------------
# CUDA source – row-wise scaling kernel
# -------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void row_scale_kernel(const float* __restrict__ A,
                                 const float* __restrict__ B,
                                 float*       __restrict__ C,
                                 int N, int M) {
    // One block per row
    const int i = blockIdx.x;

    // Load A[i] once per block
    const float a = A[i];

    // Iterate over columns with a stride equal to blockDim.x.
    // This yields coalesced accesses because threads in a warp
    // touch consecutive columns.
    int col = threadIdx.x;
    while (col < M) {
        C[i * M + col] = a * B[i * M + col];
        col += blockDim.x;
    }
}

void row_scale(at::Tensor A, at::Tensor B, at::Tensor C) {
    const int N = B.size(0);
    const int M = B.size(1);
    const int threads = 256;               // heuristic block size
    const int blocks  = N;                  // one block per row

    row_scale_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N, M
    );

    // Ensure the kernel finishes before the tensor is used
    cudaDeviceSynchronize();
}
"""

# -------------------------------------------------------------------------
# C++ binding – exposes the kernel to Python
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void row_scale(at::Tensor A, at::Tensor B, at::Tensor C);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("row_scale", &row_scale, "Row-wise scaling (A[i] * B[i,:])");
}
"""

# -------------------------------------------------------------------------
# Compile the extension
# -------------------------------------------------------------------------
row_scale_ext = load_inline(
    name="row_scale_ext",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    with_cuda=True,
)

# -------------------------------------------------------------------------
# The functional model that will be evaluated
# -------------------------------------------------------------------------
def functional_model(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs C = A[:, None] * B using a custom CUDA kernel.
    A shape: (N,)      – row scalars
    B shape: (N, M)    – matrix to be scaled row-wise
    Returns C shape: (N, M)
    """
    # Make sure inputs are on the GPU
    if not A.is_cuda:
        raise RuntimeError("A must be a CUDA tensor")
    if not B.is_cuda:
        raise RuntimeError("B must be a CUDA tensor")

    N, M = B.shape
    # Allocate output (float32 to match the kernel's float type)
    C = torch.empty((N, M), dtype=torch.float32, device=B.device)

    # Call the compiled kernel
    row_scale_ext.row_scale(A, B, C)

    return C

# -------------------------------------------------------------------------
# Dummy initialization / input generation (kept for completeness)
# -------------------------------------------------------------------------
M = 4096
N = 4096

def get_init_inputs():
    return []   # no special init

def get_inputs():
    A = torch.rand(N, device="cuda", dtype=torch.float32)
    B = torch.rand(N, M, device="cuda", dtype=torch.float32)
    return [A, B]
