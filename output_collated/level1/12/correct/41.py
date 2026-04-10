# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_102701/code_18.py
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
# Each thread handles one index of the output (N * M).
# With A of shape [N] and B of shape [N, M], index (i, j) 
# refers to A[i] * B[i * M + j] where i = idx / M, j = idx % M.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_broadcast_mul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ out,
    int N,
    int M
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * M) {
        int i = idx / M;
        // B is flattened, idx is effectively (i * M + j)
        out[idx] = A[i] * B[idx];
    }
}

void fused_broadcast_mul_forward(int blocks, int threads, torch::Tensor A, torch::Tensor B, torch::Tensor out) {
    fused_broadcast_mul_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        out.data_ptr<float>(),
        (int)A.size(0),
        (int)B.size(1)
    );
}
"""

# --- C++ Logic (Interface/Bindings) ---
cpp_source = r"""
#include <torch/extension.h>

void fused_broadcast_mul_forward(int blocks, int threads, torch::Tensor A, torch::Tensor B, torch::Tensor out);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_broadcast_mul_forward, "Fused broadcast multiply kernel");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_broadcast_mul',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(A, B):
    """
    Computes A.unsqueeze(1) * B using a custom CUDA kernel.
    A: [N]
    B: [N, M]
    Result: [N, M]
    """
    N, M = B.size()
    out = torch.empty_like(B)
    
    # Kernel configuration
    threads = 256
    blocks = (N * M + threads - 1) // threads
    
    # Launch kernel
    fused_ext.fused_op(blocks, threads, A, B, out)
    
    return out

# Constants
M = 4096
N = 4096

def get_init_inputs():
    return []

def get_inputs():
    # Ensure tensors are on GPU as required by the CUDA implementation
    A = torch.rand(N, device='cuda')
    B = torch.rand(N, M, device='cuda')
    return [A, B]
