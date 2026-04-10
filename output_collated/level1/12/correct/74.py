# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_103853/code_23.py
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

# The CUDA kernel performs element-wise multiplication A[i] * B[i, j]
# It ensures memory coalescing by aligning threads with the contiguous columns (M dimension).
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void broadcast_mul_kernel(const float* A, const float* B, float* out, int N, int M) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < N && col < M) {
        // Optimized access: row-major storage ensures coalesced reads/writes
        out[row * M + col] = A[row] * B[row * M + col];
    }
}

void broadcast_mul_cuda(torch::Tensor A, torch::Tensor B, torch::Tensor out) {
    int N = A.size(0);
    int M = B.size(1);
    
    // Choose block sizes to balance occupancy and local memory usage
    // threads.x is the dimension traversed coalesced (M)
    dim3 threads(32, 16); 
    dim3 blocks((M + threads.x - 1) / threads.x, (N + threads.y - 1) / threads.y);
    
    broadcast_mul_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(), 
        B.data_ptr<float>(), 
        out.data_ptr<float>(), 
        N, M
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void broadcast_mul_cuda(torch::Tensor A, torch::Tensor B, torch::Tensor out);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("broadcast_mul", &broadcast_mul_cuda, "Broadcasted multiplication kernel");
}
"""

# Compile the extension inline
fused_ext = load_inline(
    name='broadcast_mul_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(A, B):
    """
    Optimized broadcasted multiplication: A.unsqueeze(1) * B
    Uses a custom CUDA kernel to avoid storing the intermediate broadcasted tensor.
    """
    # Ensure inputs are on the correct device
    out = torch.empty_like(B)
    fused_ext.broadcast_mul(A, B, out)
    return out

# --- Evaluation setup ---
N, M = 4096, 4096

def get_init_inputs():
    return []

def get_inputs():
    # Return device-specific inputs as required by the custom kernel
    A = torch.rand(N, device='cuda', dtype=torch.float32)
    B = torch.rand(N, M, device='cuda', dtype=torch.float32)
    return [A, B]
