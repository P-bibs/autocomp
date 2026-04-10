# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_110950/code_1.py
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

__global__ void broadcast_mul_vectorized_kernel_opt(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ output,
    int N,
    int M
) {
    __shared__ float shared_A[TILE_SIZE];
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int elements_per_thread = 4;
    int global_thread_idx = bid * blockDim.x + tid;
    int global_elem_idx = global_thread_idx * elements_per_thread;
    
    // Process in tiles of TILE_SIZE rows
    for (int tile_start = 0; tile_start < N; tile_start += TILE_SIZE) {
        // Cooperatively load A values into shared memory
        if (tid < TILE_SIZE && (tile_start + tid) < N) {
            shared_A[tid] = A[tile_start + tid];
        }
        __syncthreads();
        
        // Calculate the range of elements for this tile
        int tile_end = min(tile_start + TILE_SIZE, N);
        int first_elem_in_tile = tile_start * M;
        int last_elem_in_tile = tile_end * M - 1;
        
        // Check if this thread has work to do in this tile
        if (global_elem_idx <= last_elem_in_tile && 
            (global_elem_idx + 3) >= first_elem_in_tile) {
            
            // Vectorized processing when we have 4 consecutive elements in same tile
            if (global_elem_idx >= first_elem_in_tile && 
                global_elem_idx + 3 <= last_elem_in_tile) {
                
                // Safe to load and store float4
                float4 b_vec = reinterpret_cast<const float4*>(&B[global_elem_idx])[0];
                float4 out_vec;
                
                int row0 = (global_elem_idx + 0) / M;
                int row1 = (global_elem_idx + 1) / M;
                int row2 = (global_elem_idx + 2) / M;
                int row3 = (global_elem_idx + 3) / M;
                
                // Get A values from shared memory
                out_vec.x = shared_A[row0 - tile_start] * b_vec.x;
                out_vec.y = shared_A[row1 - tile_start] * b_vec.y;
                out_vec.z = shared_A[row2 - tile_start] * b_vec.z;
                out_vec.w = shared_A[row3 - tile_start] * b_vec.w;
                
                reinterpret_cast<float4*>(&output[global_elem_idx])[0] = out_vec;
            } else {
                // Scalar processing for elements that cross tile boundaries
                for (int i = 0; i < elements_per_thread; i++) {
                    int elem_idx = global_elem_idx + i;
                    if (elem_idx >= first_elem_in_tile && 
                        elem_idx <= last_elem_in_tile && 
                        elem_idx < N * M) {
                        int row_idx = elem_idx / M;
                        output[elem_idx] = shared_A[row_idx - tile_start] * B[elem_idx];
                    }
                }
            }
        }
        __syncthreads();
    }
}

void broadcast_mul_forward(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& output) {
    const int N = A.size(0);
    const int M = B.size(1);
    const int total_elements = N * M;
    
    const int threads_per_block = 256;
    const int elements_per_thread = 4;
    const int blocks = (total_elements + threads_per_block * elements_per_thread - 1) / 
                       (threads_per_block * elements_per_thread);
    
    broadcast_mul_vectorized_kernel_opt<<<blocks, threads_per_block>>>(
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
