# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_102701/code_0.py
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

__global__ void broadcast_mul_vectorized_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ output,
    int N,
    int M
) {
    // Each thread handles 4 elements (1 vector float4)
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    int total_elements = N * M;
    
    if (idx + 3 < total_elements) {
        // Safe to load full float4
        float4 a_vec, b_vec, out_vec;
        
        // Load 4 consecutive elements from B
        b_vec = reinterpret_cast<const float4*>(B)[idx / 4];
        
        // Optimize A access by computing base row and checking consecutive pattern
        int base_row = idx / M;
        int next_row_1 = (idx + 1) / M;
        int next_row_2 = (idx + 2) / M;
        int next_row_3 = (idx + 3) / M;
        
        // Load A values with better memory access pattern
        if (base_row == next_row_3) {
            // All 4 elements are from the same row - single A access
            float a_val = A[base_row];
            a_vec.x = a_val;
            a_vec.y = a_val;
            a_vec.z = a_val;
            a_vec.w = a_val;
        } else {
            // Different rows - load each needed A value
            a_vec.x = A[base_row];
            a_vec.y = A[next_row_1];
            a_vec.z = A[next_row_2];
            a_vec.w = A[next_row_3];
        }
        
        // Perform element-wise multiplication
        out_vec.x = a_vec.x * b_vec.x;
        out_vec.y = a_vec.y * b_vec.y;
        out_vec.z = a_vec.z * b_vec.z;
        out_vec.w = a_vec.w * b_vec.w;
        
        // Store result
        reinterpret_cast<float4*>(output)[idx / 4] = out_vec;
    } else {
        // Handle remaining elements scalarly
        for (int i = 0; i < 4 && idx + i < total_elements; ++i) {
            int current_idx = idx + i;
            int n = current_idx / M;
            output[current_idx] = A[n] * B[current_idx];
        }
    }
}

void broadcast_mul_forward(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& output) {
    const int N = A.size(0);
    const int M = B.size(1);
    const int total_elements = N * M;
    
    const int threads_per_block = 256;
    // Divide by 4 because each thread processes 4 floats
    const int blocks = (total_elements + 4 * threads_per_block - 1) / (4 * threads_per_block);
    
    broadcast_mul_vectorized_kernel<<<blocks, threads_per_block>>>(
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
    if not A.is_cuda: A = A.cuda()
    if not B.is_cuda: B = B.cuda()
    
    output = torch.empty_like(B)
    fused_ext.broadcast_mul(A, B, output)
    return output

M = 4096
N = 4096

def get_init_inputs():
    return []  # No special initialization inputs needed

def get_inputs():
    A = torch.rand(N)
    B = torch.rand(N, M)
    return [A, B]
