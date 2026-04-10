# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_101900/code_20.py
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

# ------------------------------------------------------------------
# CUDA kernel: Grid-stride loop version of the fused broadcast multiply
# Use 1-D grid to process elements, maximizing SM occupancy and 
# hiding memory latency through persistent thread-level loops.
# ------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void fused_multiply_kernel(
        const scalar_t* __restrict__ A,
        const scalar_t* __restrict__ B,
        scalar_t* __restrict__ Out,
        const int64_t total_elements,
        const int64_t M)
{
    const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = blockDim.x * gridDim.x;

    for (int64_t idx = tid; idx < total_elements; idx += stride) {
        const int64_t row = idx / M;
        Out[idx] = A[row] * B[idx];
    }
}

void fused_op_forward(torch::Tensor A, torch::Tensor B, torch::Tensor Out) {
    const int64_t N = A.size(0);
    const int64_t M = B.size(1);
    const int64_t total_elements = N * M;

    // Use a fixed block size; grid size is determined by hardware capacity
    const int threads_per_block = 256;
    // We cap blocks to avoid excessive launch overhead, allowing stride to cover the work
    const int max_blocks = 1024; 
    const int num_blocks = (int)std::min((int64_t)max_blocks, (total_elements + threads_per_block - 1) / threads_per_block);

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "fused_multiply_kernel", ([&] {
        fused_multiply_kernel<scalar_t><<<num_blocks, threads_per_block>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            Out.data_ptr<scalar_t>(),
            total_elements,
            M);
    }));
}
"""

# ------------------------------------------------------------------
# C++ binding
# ------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>
void fused_op_forward(torch::Tensor A, torch::Tensor B, torch::Tensor Out);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused grid-stride broadcast multiply");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_mul_gridstride',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(A: torch.Tensor, B: torch.Tensor):
    """
    Optimized functional model using the grid-stride kernel.
    Input A is [N], B is [N, M].
    Output [N, M] = A[row] * B[row, col].
    """
    N, M = B.shape
    out = torch.empty((N, M), device=A.device, dtype=A.dtype)
    fused_ext.fused_op(A, B, out)
    return out

# --- Setup for benchmarking compatibility ---
M, N = 4096, 4096

def get_init_inputs():
    return []

def get_inputs():
    # Ensure tensors are on GPU as per requirement
    A = torch.rand(N, device='cuda')
    B = torch.rand(N, M, device='cuda')
    return [A, B]
