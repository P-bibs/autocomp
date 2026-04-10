# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_105940/code_17.py
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

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void broadcast_mul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ output,
    int N,
    int M
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Process 4 floats per thread using float4 for memory coalescing
    int total_float4 = (N * M) / 4;
    
    for (int i = idx; i < total_float4; i += stride) {
        int base_idx = i * 4;
        float4 b_val = reinterpret_cast<const float4*>(B)[i];
        
        // Calculate row indices based on current flat index
        int row0 = base_idx / M;
        int row1 = (base_idx + 1) / M;
        int row2 = (base_idx + 2) / M;
        int row3 = (base_idx + 3) / M;
        
        float4 out_val;
        out_val.x = b_val.x * A[row0];
        out_val.y = b_val.y * A[row1];
        out_val.z = b_val.z * A[row2];
        out_val.w = b_val.w * A[row3];
        
        reinterpret_cast<float4*>(output)[i] = out_val;
    }
    
    // Handle remainder
    int remainder_start = total_float4 * 4;
    for (int i = remainder_start + idx; i < N * M; i += stride) {
        output[i] = B[i] * A[i / M];
    }
}

void broadcast_mul_forward(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& output) {
    const int N = A.size(0);
    const int M = B.size(1);
    const int total_elements = N * M;
    
    const int threads = 256;
    const int blocks = (total_elements / 4 + threads - 1) / threads;
    
    broadcast_mul_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        output.data_ptr<float>(),
        N,
        M
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void broadcast_mul_forward(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& output);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("broadcast_mul", &broadcast_mul_forward, "Vectorized broadcast multiplication");
}
"""

fused_ext = load_inline(
    name='broadcast_mul_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(A, B):
    # Ensure inputs are contiguous and on correct device
    A = A.contiguous().cuda()
    B = B.contiguous().cuda()
    
    output = torch.empty_like(B)
    fused_ext.broadcast_mul(A, B, output)
    return output

M = 4096
N = 4096

def get_init_inputs():
    return []

def get_inputs():
    A = torch.rand(N)
    B = torch.rand(N, M)
    return [A, B]
