# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_101900/code_22.py
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
# CUDA kernel – one block per row, loads A[row] only once per block into 
# registers, then performs fully coalesced element‑wise multiplication with B.
# This eliminates global memory pressure on A and removes index arithmetic.
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
    // Each block handles one whole row of the matrix B.
    // The value A[row] is unique to the entire block.
    int row = blockIdx.x;
    
    // Load the broadcast scalar A[row] into a register for this block
    float a_val = __ldg(A + row);

    // Cooperative processing: each thread processes multiple columns
    // to ensure full coalesced access pattern for memory B.
    int col = threadIdx.x;
    int stride = blockDim.x;

    for (int m = col; m < M; m += stride) {
        int idx = row * M + m;
        // __ldg ensures read-only cache usage for global matrix B
        float b_val = __ldg(B + idx);
        output[idx] = a_val * b_val;
    }
}

void broadcast_mul_forward(
    const torch::Tensor& A,
    const torch::Tensor& B,
    torch::Tensor& output
) {
    const int N = A.size(0);
    const int M = B.size(1);

    // 256 threads per block is generally optimal for memory-bound ops
    const int threads = 256;
    const int blocks = N;

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
# C++ interface (pybind11)
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void broadcast_mul_forward(
    const torch::Tensor& A,
    const torch::Tensor& B,
    torch::Tensor& output
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("broadcast_mul", &broadcast_mul_forward, "Optimized broadcast multiplication kernel");
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
# Functional wrapper
# -------------------------------------------------------------------------
def functional_model(A, B):
    # Ensure inputs are on the GPU; the kernel expects contiguous tensors.
    # The evaluation assumes A has shape (N,) and B has shape (N, M).
    if not A.is_cuda:
        A = A.cuda()
    if not B.is_cuda:
        B = B.cuda()
    
    # Ensure inputs are contiguous to satisfy __ldg and coalesced access requirements
    if not A.is_contiguous():
        A = A.contiguous()
    if not B.is_contiguous():
        B = B.contiguous()
    
    output = torch.empty_like(B)
    
    # Launch custom kernel
    fused_ext.broadcast_mul(A, B, output)
    
    return output

# Parameters for testing
M = 4096
N = 4096

def get_init_inputs():
    return []

def get_inputs():
    A = torch.rand(N).cuda()
    B = torch.rand(N, M).cuda()
    return [A, B]
