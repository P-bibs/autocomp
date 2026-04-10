# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_103853/code_31.py
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
# CUDA kernel and host wrapper (compiled on-the-fly)
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector_types.h>

// Kernel: scale each row of B by the corresponding element of A.
// Each block handles one row. Threads process the row using float4 for vectorized memory access.
__global__ void scale_row_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float*       __restrict__ C,
    int M)
{
    // Scalar a for this row
    __shared__ float a;
    if (threadIdx.x == 0) {
        a = A[blockIdx.x];
    }
    __syncthreads();

    float local_a = a;
    int row_offset = blockIdx.x * M;

    // Vectorized processing: 4 elements at a time
    int idx = threadIdx.x * 4;
    for (; idx <= M - 4; idx += blockDim.x * 4) {
        float4 b_vec = reinterpret_cast<const float4*>(&B[row_offset + idx])[0];
        float4 c_vec;
        c_vec.x = b_vec.x * local_a;
        c_vec.y = b_vec.y * local_a;
        c_vec.z = b_vec.z * local_a;
        c_vec.w = b_vec.w * local_a;
        reinterpret_cast<float4*>(&C[row_offset + idx])[0] = c_vec;
    }

    // Cleanup for non-multiple of 4 dimensions
    for (int i = idx; i < M; ++i) {
        C[row_offset + i] = B[row_offset + i] * local_a;
    }
}

void scale_rows(int N, int M, torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    // 128 threads per block to balance occupancy and performance
    const int threads = 128;
    const int blocks = N;
    scale_row_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(), 
        B.data_ptr<float>(), 
        C.data_ptr<float>(), 
        M
    );
}
"""

# ----------------------------------------------------------------------
# C++ bindings (pybind11)
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void scale_rows(int N, int M, torch::Tensor A, torch::Tensor B, torch::Tensor C);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("scale_rows", &scale_rows, "Scale matrix B rows by vector A");
}
"""

# Compile the extension
ext = load_inline(
    name='scale_rows_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Optimized broadcasted multiplication using a custom fused CUDA kernel.
    """
    # Ensure inputs are on GPU
    A = A.to(device='cuda', dtype=torch.float32).contiguous()
    B = B.to(device='cuda', dtype=torch.float32).contiguous()
    
    N, M = B.shape
    C = torch.empty_like(B)
    
    # Launch CUDA kernel
    ext.scale_rows(N, M, A, B, C)
    
    return C

def get_init_inputs():
    return []

def get_inputs():
    M = 4096
    N = 4096
    A = torch.rand(N, dtype=torch.float32)
    B = torch.rand(N, M, dtype=torch.float32)
    return [A, B]
