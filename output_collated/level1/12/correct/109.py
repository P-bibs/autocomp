# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_105940/code_1.py
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

#define TILE_SIZE 256

__global__ void broadcast_mul_vectorized_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ output,
    int N,
    int M
) {
    __shared__ float shared_A[TILE_SIZE];
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int total_elements = N * M;
    
    // Each thread processes 4 elements (1 float4)
    int idx = (bid * blockDim.x + tid) * 4;
    
    // Load shared memory with A values for this tile
    int rows_start = bid * TILE_SIZE;
    int rows_end = min(rows_start + TILE_SIZE, N);
    
    // Coalesced load of A into shared memory
    for (int i = tid; i < rows_end - rows_start; i += blockDim.x) {
        shared_A[i] = A[rows_start + i];
    }
    __syncthreads();
    
    if (idx + 3 < total_elements) {
        float4 a_vec, b_vec, out_vec;
        
        // Load 4 consecutive elements from B (coalesced access)
        b_vec = reinterpret_cast<const float4*>(B)[idx / 4];
        
        // Compute row indices for the 4 elements
        int n0 = idx / M;
        int n1 = (idx + 1) / M;
        int n2 = (idx + 2) / M;
        int n3 = (idx + 3) / M;
        
        // Load A values from shared memory when possible
        int local_n0 = n0 - rows_start;
        int local_n1 = n1 - rows_start;
        int local_n2 = n2 - rows_start;
        int local_n3 = n3 - rows_start;
        
        // Check bounds and load from shared memory or global memory
        a_vec.x = (local_n0 >= 0 && local_n0 < rows_end - rows_start) ? 
                  shared_A[local_n0] : A[n0];
        a_vec.y = (local_n1 >= 0 && local_n1 < rows_end - rows_start) ? 
                  shared_A[local_n1] : A[n1];
        a_vec.z = (local_n2 >= 0 && local_n2 < rows_end - rows_start) ? 
                  shared_A[local_n2] : A[n2];
        a_vec.w = (local_n3 >= 0 && local_n3 < rows_end - rows_start) ? 
                  shared_A[local_n3] : A[n3];
        
        // Perform element-wise multiplication
        out_vec.x = a_vec.x * b_vec.x;
        out_vec.y = a_vec.y * b_vec.y;
        out_vec.z = a_vec.z * b_vec.z;
        out_vec.w = a_vec.w * b_vec.w;
        
        // Store result (coalesced write)
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
    m.def("broadcast_mul", &broadcast_mul_forward, "Vectorized broadcast multiplication with shared memory");
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
