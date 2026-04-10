# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_101900/code_15.py
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

# The CUDA kernel performs the element-wise multiplication A[i] * B[i, j].
# Since A[i] is accessed for every column j, it stays in register/L1 cache.
# B is accessed in a coalesced manner, ensuring high memory bandwidth utilization.
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_op_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ out, int N, int M) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < N && j < M) {
        // Broadcast A[i] across row i of B
        out[i * M + j] = A[i] * B[i * M + j];
    }
}

void fused_op_forward(torch::Tensor A, torch::Tensor B, torch::Tensor out) {
    int N = A.size(0);
    int M = B.size(1);
    
    // Using a 2D block size that is powers of 2 for optimal warp occupancy
    dim3 threads(32, 16);
    dim3 blocks((M + threads.x - 1) / threads.x, (N + threads.y - 1) / threads.y);
    
    fused_op_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(), 
        B.data_ptr<float>(), 
        out.data_ptr<float>(), 
        N, M
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(torch::Tensor A, torch::Tensor B, torch::Tensor out);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused multiplication kernel");
}
"""

# Compile the extension inline
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(A, B):
    """
    Optimized implementation of A.unsqueeze(1) * B using a fused CUDA kernel.
    """
    N, M = B.shape
    # Ensure inputs are contiguous to optimize memory access patterns
    A = A.contiguous()
    B = B.contiguous()
    out = torch.empty((N, M), device=A.device, dtype=A.dtype)
    
    fused_ext.fused_op(A, B, out)
    return out

def get_init_inputs():
    return []

def get_inputs():
    N, M = 4096, 4096
    A = torch.rand(N, device='cuda', dtype=torch.float32)
    B = torch.rand(N, M, device='cuda', dtype=torch.float32)
    return [A, B]
