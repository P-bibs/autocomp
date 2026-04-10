# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_112116/code_7.py
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

# Optimization: Coalesced memory access and explicit kernel fusion
# The kernel maps (i, j) to the output grid, ensuring coalesced access to row-major B.
cuda_kernel = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void broadcast_mul_kernel(const float* A, const float* B, float* out, int N, int M) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < N && j < M) {
        // A is accessed once per row, B and out are accessed with coalesced patterns
        out[i * M + j] = A[i] * B[i * M + j];
    }
}

void fused_op_forward_kernelLauncher(const float* A, const float* B, float* out, int N, int M) {
    dim3 threads(32, 8); // 256 threads
    dim3 blocks((M + threads.x - 1) / threads.x, (N + threads.y - 1) / threads.y);
    
    broadcast_mul_kernel<<<blocks, threads>>>(
        A, B, out, N, M);
}
'''

cpp_source = r'''
#include <torch/extension.h>

void fused_op_forward_kernelLauncher(const float* A, const float* B, float* out, int N, int M);

void fused_op_forward(torch::Tensor A, torch::Tensor B, torch::Tensor out) {
    int N = A.size(0);
    int M = B.size(1);
    fused_op_forward_kernelLauncher(A.data_ptr<float>(), B.data_ptr<float>(), out.data_ptr<float>(), N, M);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused broadcast multiplication kernel");
}
'''

fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(A, B):
    N, M = B.shape
    out = torch.empty((N, M), device=A.device, dtype=A.dtype)
    fused_ext.fused_op(A, B, out)
    return out

# Initialization for the evaluation context
M = 4096
N = 4096

def get_init_inputs():
    return []

def get_inputs():
    A = torch.rand(N, device='cuda', dtype=torch.float32)
    B = torch.rand(N, M, device='cuda', dtype=torch.float32)
    return [A, B]
