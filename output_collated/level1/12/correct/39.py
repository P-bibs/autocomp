# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_102701/code_16.py
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

// Optimized kernel using 1D grid mapping to the matrix B elements.
// Since A is broadcasted, we access A[idx / M].
__global__ void broadcast_mul_vectorized_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ output,
    int N,
    int M
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * M;

    // Use float4 for vectorized memory access to B and output
    if (idx * 4 + 3 < total_elements) {
        int base_idx = idx * 4;
        
        // Load one vector from B
        float4 b_vec = reinterpret_cast<const float4*>(B)[idx];
        
        // Broadcasted values from A 
        // Note: For large M, idx / M changes infrequently.
        // Reading A is cached in L1/Texture cache.
        float4 a_vec;
        a_vec.x = A[base_idx / M];
        a_vec.y = A[(base_idx + 1) / M];
        a_vec.z = A[(base_idx + 2) / M];
        a_vec.w = A[(base_idx + 3) / M];
        
        float4 out_vec = {
            a_vec.x * b_vec.x,
            a_vec.y * b_vec.y,
            a_vec.z * b_vec.z,
            a_vec.w * b_vec.w
        };
        
        reinterpret_cast<float4*>(output)[idx] = out_vec;
    } else {
        // Scalar fallback for boundary
        for (int i = idx * 4; i < total_elements; ++i) {
            output[i] = A[i / M] * B[i];
        }
    }
}

void broadcast_mul_forward(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& output) {
    const int N = A.size(0);
    const int M = B.size(1);
    const int total_elements = N * M;
    
    // Each thread processes 4 elements
    const int threads = 256;
    const int blocks = (total_elements / 4 + threads - 1) / threads;
    
    broadcast_mul_vectorized_kernel<<<blocks, threads>>>(
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
    # Ensure inputs are contiguous for efficient float4 access
    if not A.is_contiguous(): A = A.contiguous()
    if not B.is_contiguous(): B = B.contiguous()
    if not A.is_cuda: A = A.cuda()
    if not B.is_cuda: B = B.cuda()
    
    output = torch.empty_like(B)
    fused_ext.broadcast_mul(A, B, output)
    return output

def get_inputs():
    N, M = 4096, 4096
    A = torch.rand(N)
    B = torch.rand(N, M)
    return [A, B]
