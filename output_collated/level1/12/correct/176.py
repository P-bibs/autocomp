# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_112116/code_19.py
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

# CUDA kernel with coalesced memory access
# Each thread handles a unique index in the flattened N*M matrix
cuda_source = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void elementwise_mul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ out,
    const int N,
    const int M
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * M;
    
    if (idx < total_elements) {
        // Since B is (N, M), the row index is idx / M
        // A is accessed via broadcasting: A[idx / M]
        out[idx] = A[idx / M] * B[idx];
    }
}

void launch_elementwise_mul(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& out) {
    const int N = A.size(0);
    const int M = B.size(1);
    const int total_threads = N * M;
    
    const int threads = 256;
    const int blocks = (total_threads + threads - 1) / threads;
    
    elementwise_mul_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        out.data_ptr<float>(),
        N,
        M
    );
}
'''

cpp_source = r'''
#include <torch/extension.h>
void launch_elementwise_mul(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& out);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &launch_elementwise_mul, "Elementwise broadcasted multiplication");
}
'''

# Compile the JIT extension
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(A, B):
    # Ensure inputs are contiguous, which is required for efficient coalesced indexing
    A = A.contiguous().cuda()
    B = B.contiguous().cuda()
    output = torch.empty_like(B)
    
    fused_ext.fused_op(A, B, output)
    return output

N, M = 4096, 4096

def get_init_inputs():
    return []

def get_inputs():
    return [torch.rand(N).cuda(), torch.rand(N, M).cuda()]
