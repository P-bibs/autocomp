# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_112116/code_1.py
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
    // Shared memory to cache a portion of vector A
    extern __shared__ float shared_A[];
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int threads_per_block = blockDim.x;
    
    // Each thread handles 4 elements (1 vector float4)
    const int elements_per_thread = 4;
    const int total_elements = N * M;
    const int elements_per_block = threads_per_block * elements_per_thread;
    
    // Grid-stride loop to handle cases where we have more elements than blocks can handle
    for (int block_start = bid * elements_per_block; 
         block_start < total_elements; 
         block_start += gridDim.x * elements_per_block) {
        
        // Calculate which rows of A are needed for this block
        int first_row = block_start / M;
        int last_row = min((block_start + elements_per_block - 1) / M, N - 1);
        int rows_needed = last_row - first_row + 1;
        
        // Load the required portion of A into shared memory cooperatively
        for (int i = tid; i < rows_needed; i += threads_per_block) {
            shared_A[i] = A[first_row + i];
        }
        __syncthreads();
        
        // Process elements in this block
        int idx = block_start + tid * elements_per_thread;
        
        if (idx + 3 < min(block_start + elements_per_block, total_elements)) {
            // Safe to load full float4
            float4 b_vec = reinterpret_cast<const float4*>(&B[idx])[0];
            float4 out_vec;
            
            // Compute row indices for the 4 elements
            int n0 = idx / M;
            int n1 = (idx + 1) / M;
            int n2 = (idx + 2) / M;
            int n3 = (idx + 3) / M;
            
            // Load from shared memory (A values are now offset by first_row)
            out_vec.x = shared_A[n0 - first_row] * b_vec.x;
            out_vec.y = shared_A[n1 - first_row] * b_vec.y;
            out_vec.z = shared_A[n2 - first_row] * b_vec.z;
            out_vec.w = shared_A[n3 - first_row] * b_vec.w;
            
            reinterpret_cast<float4*>(&output[idx])[0] = out_vec;
        } else {
            // Handle remaining elements scalarly
            for (int i = 0; i < elements_per_thread && idx + i < total_elements; ++i) {
                int current_idx = idx + i;
                if (current_idx < total_elements) {
                    int n = current_idx / M;
                    output[current_idx] = shared_A[n - first_row] * B[current_idx];
                }
            }
        }
        __syncthreads(); // Ensure all threads are done with shared memory before next iteration
    }
}

void broadcast_mul_forward(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& output) {
    const int N = A.size(0);
    const int M = B.size(1);
    const int total_elements = N * M;
    
    const int threads_per_block = 256;
    const int elements_per_block = threads_per_block * 4; // 4 elements per thread
    
    // Calculate shared memory size needed (maximum possible rows in a block)
    const int max_rows_per_block = (elements_per_block + M - 1) / M + 1;
    const int shared_mem_size = max_rows_per_block * sizeof(float);
    
    // Use a reasonable grid size for the RTX 2080Ti
    const int blocks = min(65535, (total_elements + elements_per_block - 1) / elements_per_block);
    
    broadcast_mul_vectorized_kernel<<<blocks, threads_per_block, shared_mem_size>>>(
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
    m.def("broadcast_mul", &broadcast_mul_forward, "Optimized broadcast multiplication with shared memory");
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
