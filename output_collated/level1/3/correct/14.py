# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_091341/code_0.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = []
FORWARD_ARG_NAMES = ['A', 'B']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = []
REQUIRED_FLAT_STATE_NAMES = []


class ModelNew(nn.Module):
    """
    Performs batched matrix multiplication (C = A * B) where A, B, and C have the same batch dimension.
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

# Create custom CUDA kernel for batch matrix multiplication
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void batch_matmul_kernel(
    const float* A,
    const float* B,
    float* C,
    int batch_size,
    int m,
    int n,
    int k
) {
    int batch_idx = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m && col < n && batch_idx < batch_size) {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            sum += A[batch_idx * m * k + row * k + i] * B[batch_idx * k * n + i * n + col];
        }
        C[batch_idx * m * n + row * n + col] = sum;
    }
}

void batch_matmul_forward(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C,
    int batch_size,
    int m,
    int n,
    int k
) {
    dim3 block(16, 16);
    dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y, batch_size);
    
    batch_matmul_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        batch_size,
        m,
        n,
        k
    );
}
"""

# C++ interface/bindings
cpp_source = r"""
#include <torch/extension.h>

void batch_matmul_forward(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C,
    int batch_size,
    int m,
    int n,
    int k
);

torch::Tensor fused_batch_matmul(torch::Tensor A, torch::Tensor B) {
    int batch_size = A.size(0);
    int m = A.size(1);
    int k = A.size(2);
    int n = B.size(2);
    
    auto C = torch::zeros({batch_size, m, n}, A.options());
    
    batch_matmul_forward(A, B, C, batch_size, m, n, k);
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_batch_matmul", &fused_batch_matmul, "Batch matrix multiplication with optimizations");
}
"""

# Compile the extension with optimization flags
optimized_matmul = load_inline(
    name='optimized_matmul',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(A, B):
    return optimized_matmul.fused_batch_matmul(A, B)

batch_size = 128
m = 128 * 4
k = 256 * 4
n = 512 * 4

def get_init_inputs():
    return []  # No special initialization inputs needed

def get_inputs():
    A = torch.rand(batch_size, m, k, device='cuda')
    B = torch.rand(batch_size, k, n, device='cuda')
    return [A, B]
