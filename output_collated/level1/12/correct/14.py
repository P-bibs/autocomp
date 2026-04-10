# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_101900/code_10.py
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# -------------------------------------------------------------------------
# CUDA kernel – one block per row, loads A[row] only once per block,
# then performs coalesced element‑wise multiplication with B.
# -------------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void broadcast_mul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ output,
    int N,
    int M
) {
    // Shared memory to hold the single A value for the current row
    __shared__ float a_val;

    // Each block handles one whole row of the output
    int row = blockIdx.x;          // row index == block index
    int col = threadIdx.x;         // initial column offset inside the row
    int stride = blockDim.x;       // number of threads per block

    // Load A[row] once per block (thread 0 does the load)
    if (threadIdx.x == 0) {
        a_val = __ldg(A + row);
    }
    __syncthreads();

    float a = a_val;   // broadcast to all threads in the block

    // Iterate over the columns of this row with a stride equal to blockDim.x
    for (int m = col; m < M; m += stride) {
        int idx = row * M + m;                // linear index in row‑major storage
        float b = __ldg(B + idx);             // read‑only cache load
        output[idx] = a * b;                  // element‑wise multiply
    }
}

void broadcast_mul_forward(
    const torch::Tensor& A,
    const torch::Tensor& B,
    torch::Tensor& output
) {
    const int N = A.size(0);
    const int M = B.size(1);

    const int threads = 256;               // block dimension (multiple of 32)
    const int blocks  = N;                  // one block per row

    broadcast_mul_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        output.data_ptr<float>(),
        N,
        M
    );
}
"""

# -------------------------------------------------------------------------
# C++ interface (pybind11) – exposes the CUDA function to Python
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void broadcast_mul_forward(
    const torch::Tensor& A,
    const torch::Tensor& B,
    torch::Tensor& output
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("broadcast_mul", &broadcast_mul_forward,
          "Broadcast multiplication kernel (CUDA)");
}
"""

# Compile the inline extension
fused_ext = load_inline(
    name='broadcast_mul_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# -------------------------------------------------------------------------
# Functional wrapper – the entry point used by the evaluator
# -------------------------------------------------------------------------
def functional_model(A, B):
    # Ensure inputs reside on the GPU
    if not A.is_cuda:
        A = A.cuda()
    if not B.is_cuda:
        B = B.cuda()

    # Allocate output tensor with the same shape and dtype as B
    output = torch.empty_like(B)

    # Invoke the optimised CUDA kernel
    fused_ext.broadcast_mul(A, B, output)

    return output

# -------------------------------------------------------------------------
# Helper functions required by the benchmark harness
# -------------------------------------------------------------------------
M = 4096
N = 4096

def get_init_inputs():
    # No special initialisation needed
    return []

def get_inputs():
    A = torch.rand(N)          # shape (N,)
    B = torch.rand(N, M)       # shape (N, M)
    return [A, B]
