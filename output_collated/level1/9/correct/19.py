# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_100031/code_5.py
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

# The CUDA kernel uses a tiled approach. 
# M is large (32768), N is small (32).
# We optimize global memory access by ensuring threads in a warp access 
# contiguous memory addresses when writing the result C.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void matmul_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < M) {
        float sum = 0.0f;
        // The inner loop over N (32) is small, unrolling it allows for better scheduling
        #pragma unroll
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * M + col];
        }
        C[row * M + col] = sum;
    }
}

torch::Tensor matmul(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int N = A.size(1);
    auto C = torch::zeros({M, M}, A.options());
    
    // Using 16x16 tiles provides good occupancy and maps well to warp sizes
    dim3 threads(16, 16);
    dim3 blocks((M + 15) / 16, (M + 15) / 16);
    
    matmul_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N
    );
    return C;
}
"""

cpp_source = r"""
#include <torch/extension.h>
torch::Tensor matmul(torch::Tensor A, torch::Tensor B);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul", &matmul, "Optimized coalesced matrix multiplication");
}
"""

# Compile the custom kernel
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(A, B):
    # Call the custom CUDA extension
    return fused_ext.matmul(A, B)

M = 16384 * 2
N = 16 * 2

def get_init_inputs():
    return []

def get_inputs():
    # Use CUDA device as required for the kernel
    A = torch.rand(M, N, device='cuda', dtype=torch.float32)
    B = torch.rand(N, M, device='cuda', dtype=torch.float32)
    return [A, B]
