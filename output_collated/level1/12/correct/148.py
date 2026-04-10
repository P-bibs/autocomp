# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_110950/code_19.py
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

# --- CUDA Kernel ---
# The logic C[i, j] = A[i] * B[i, j] is implemented.
# Memory access is coalesced because thread i processes consecutive elements of B and C.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void broadcast_mul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int N,
    const int M
) {
    long long total_elements = (long long)N * M;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (long long i = idx; i < total_elements; i += stride) {
        int n_idx = (int)(i / M);
        C[i] = A[n_idx] * B[i];
    }
}

void broadcast_mul_forward(
    const torch::Tensor& A,
    const torch::Tensor& B,
    torch::Tensor& C
) {
    const int N = A.size(0);
    const int M = B.size(1);
    const int threads = 256;
    const long long total_elements = (long long)N * M;
    const int blocks = (int)std::min((total_elements + threads - 1) / threads, 65535LL);
    
    broadcast_mul_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N,
        M
    );
}
"""

# --- C++ Logic (Interface/Bindings) ---
cpp_source = r"""
#include <torch/extension.h>

void broadcast_mul_forward(
    const torch::Tensor& A,
    const torch::Tensor& B,
    torch::Tensor& C
);

torch::Tensor fused_broadcast_mul(torch::Tensor A, torch::Tensor B) {
    auto C = torch::empty_like(B);
    broadcast_mul_forward(A, B, C);
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_broadcast_mul", &fused_broadcast_mul, "Fused broadcast multiplication kernel");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_broadcast_mul_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(A, B):
    """
    Computes A.unsqueeze(1) * B using a custom CUDA fused kernel.
    """
    return fused_ext.fused_broadcast_mul(A, B)

# Constants used for the original problem scope
M = 4096
N = 4096

def get_init_inputs():
    return []

def get_inputs():
    # Devices must be CUDA for the fused kernel to operate
    A = torch.rand(N, device='cuda', dtype=torch.float32)
    B = torch.rand(N, M, device='cuda', dtype=torch.float32)
    return [A, B]
