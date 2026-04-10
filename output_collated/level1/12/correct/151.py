# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_110950/code_23.py
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

# The CUDA kernel performs element-wise broadcasting: result[i, j] = A[i] * B[i, j]
# Each thread handles a row index, and processes columns using float4 vectorization 
# to ensure memory coalescing and maximize bandwidth utilization on the 2080Ti.
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void broadcast_mul_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ out, int N, int M) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N) {
        float a_val = A[row];
        int row_offset = row * M;
        
        // Use reinterpret_cast for float4 vectorization to maximize throughput
        const float4* b_ptr = reinterpret_cast<const float4*>(&B[row_offset]);
        float4* out_ptr = reinterpret_cast<float4*>(&out[row_offset]);
        
        int vec_M = M / 4;
        for (int i = 0; i < vec_M; ++i) {
            float4 b = b_ptr[i];
            b.x *= a_val;
            b.y *= a_val;
            b.z *= a_val;
            b.w *= a_val;
            out_ptr[i] = b;
        }
    }
}

torch::Tensor broadcast_mul(torch::Tensor A, torch::Tensor B) {
    auto out = torch::empty_like(B);
    int N = A.size(0);
    int M = B.size(1);
    
    // Grid/Block configuration
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    
    broadcast_mul_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(), 
        B.data_ptr<float>(), 
        out.data_ptr<float>(), 
        N, M
    );
    
    return out;
}
"""

cpp_source = r"""
#include <torch/extension.h>

torch::Tensor broadcast_mul(torch::Tensor A, torch::Tensor B);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("broadcast_mul", &broadcast_mul, "Optimized broadcasting multiplication");
}
"""

# Compile the JIT extension
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(A, B):
    """
    Optimized broadcasting multiplication A.unsqueeze(1) * B.
    Expects A (N) and B (N, M) on CUDA.
    """
    return fused_ext.broadcast_mul(A, B)

# Helper functions required by the prompt structure
def get_init_inputs():
    return []

def get_inputs():
    N, M = 4096, 4096
    A = torch.rand(N).cuda()
    B = torch.rand(N, M).cuda()
    return [A, B]
