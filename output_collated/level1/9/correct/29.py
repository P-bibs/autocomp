# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_100811/code_7.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = []
FORWARD_ARG_NAMES = ['A', 'B']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = []
REQUIRED_FLAT_STATE_NAMES = []


class ModelNew(nn.Module):
    """
    Simple model that performs a single matrix multiplication (C = A * B) where one of the matrices is tall and skinny (M >> N or N >> M)
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

# Problem size
M = 16384 * 2   # 32768
N = 16 * 2      # 32

# ------------------- CUDA kernel source -------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// Each thread computes one element C[i,j] = sum(A[i,k] * B[k,j])
// M=32768, N=32. Threads read contiguous A, strided B.
__global__ void matmul_kernel(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float* __restrict__ C,
                              int M,
                              int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < M && j < M) {
        float sum = 0.0f;
        int row_offset = i * N;
        // Small N makes this loop efficient; register usage is low.
        #pragma unroll
        for (int k = 0; k < N; ++k) {
            sum += A[row_offset + k] * B[k * M + j];
        }
        C[i * M + j] = sum;
    }
}

void matmul_cuda(torch::Tensor A, torch::Tensor B, torch::Tensor C)
{
    const int M = A.size(0);
    const int N = A.size(1);

    // 16x16 threads per block
    dim3 block(16, 16);
    dim3 grid((M + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    matmul_kernel<<<grid, block>>>(
        A.data_ptr<float>(), 
        B.data_ptr<float>(), 
        C.data_ptr<float>(), 
        M, N);
}
"""

cpp_source = r"""
#include <torch/extension.h>

void matmul_cuda(torch::Tensor A, torch::Tensor B, torch::Tensor C);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_cuda", &matmul_cuda, "Custom CUDA matrix multiplication");
}
"""

# Compile extension
fused_ext = load_inline(
    name='matmul_cuda_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(A, B):
    device = torch.device('cuda')
    A = A.to(device=device, dtype=torch.float32)
    B = B.to(device=device, dtype=torch.float32)
    
    C = torch.empty((M, M), device=device, dtype=torch.float32)
    
    fused_ext.matmul_cuda(A, B, C)
    return C

def get_init_inputs():
    return []

def get_inputs():
    # Return CPU tensors; they are moved to GPU inside functional_model
    A = torch.rand(M, N)
    B = torch.rand(N, M)
    return [A, B]
