# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_101548/code_6.py
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

# 1. CUDA Kernel for A[i] * B[i, j]
cuda_source = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void broadcast_mul_kernel(const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float* __restrict__ Out,
                                     const int N,
                                     const int M)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < N && j < M) {
        Out[i * M + j] = A[i] * B[i * M + j];
    }
}

void fused_mul(torch::Tensor A, torch::Tensor B, torch::Tensor Out)
{
    const int N = A.size(0);
    const int M = B.size(1);

    // Using 32x32 blocks to optimize occupancy on RTX 2080 Ti
    const dim3 block(32, 32);
    const dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    broadcast_mul_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        Out.data_ptr<float>(),
        N,
        M
    );
}
'''

# 2. C++ Binding Logic
cpp_source = r'''
#include <torch/extension.h>

void fused_mul(torch::Tensor A, torch::Tensor B, torch::Tensor Out);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_mul", &fused_mul, "Fused broadcast multiplication kernel");
}
'''

# 3. Compilation
fused_ext = load_inline(
    name='fused_mul_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True,
)

# 4. Helper and Wrapper
N = 4096
M = 4096

def get_init_inputs():
    return []

def get_inputs():
    A = torch.rand(N)
    B = torch.rand(N, M)
    return [A, B]

def functional_model(A, B):
    """
    Optimized GPU implementation of A.unsqueeze(1) * B.
    """
    # Move to GPU and ensure contiguous layout for memory coherence
    A_gpu = A.contiguous().cuda()
    B_gpu = B.contiguous().cuda()
    
    out_gpu = torch.empty_like(B_gpu)
    
    # Launch the custom CUDA kernel
    fused_ext.fused_mul(A_gpu, B_gpu, out_gpu)
    
    return out_gpu
