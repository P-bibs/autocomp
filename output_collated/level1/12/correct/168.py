# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_112116/code_9.py
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

__global__ void broadcast_mul_shared_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ O,
    int N,
    int M
) {
    // ---- 1. 4-float vectorisation ------------------------------------------------
    const int vec_len = 4;
    const int thread_id = threadIdx.x;
    const int block_id = blockIdx.x;
    const int threads_per_block = blockDim.x;

    // Global linear index of the first element this thread will handle
    long long base_idx = ((long long)block_id * threads_per_block + thread_id) * vec_len;

    // ---- 2. Shared-memory tile for A --------------------------------------------
    extern __shared__ float sh_A[];

    // ---- 3. Grid-stride loop over all elements -----------------------------------
    const long long total_elems = (long long)N * M;

    for (long long idx = base_idx; idx < total_elems; idx += (long long)threads_per_block * vec_len * gridDim.x) {
        // Compute the row (n) for the first element of this vector
        int n = idx / M;
        int col_offset = idx % M;

        // Only the first thread loads A[n] into shared memory
        if (thread_id == 0) {
            sh_A[0] = A[n];
        }
        __syncthreads();

        // Load 4 floats of B as a float4 (coalesced)
        float4 b_vec = reinterpret_cast<const float4*>(B)[idx / vec_len];

        // Multiply with the cached broadcast scalar
        float a_val = sh_A[0];
        float4 out_vec;
        out_vec.x = a_val * b_vec.x;
        out_vec.y = a_val * b_vec.y;
        out_vec.z = a_val * b_vec.z;
        out_vec.w = a_val * b_vec.w;

        // Write result back (coalesced)
        reinterpret_cast<float4*>(O)[idx / vec_len] = out_vec;
        
        __syncthreads(); // Ensure no warp reuses sh_A before next iteration is safe
    }
}

void broadcast_mul_forward(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& output) {
    const int N = A.size(0);
    const int M = B.size(1);
    const long long total_elements = (long long)N * M;
    
    const int threads_per_block = 256;
    const int blocks = (total_elements + threads_per_block * 4 - 1) / (threads_per_block * 4);

    // One float of shared memory per block
    const int shared_mem_bytes = sizeof(float);

    broadcast_mul_shared_kernel<<<blocks, threads_per_block, shared_mem_bytes>>>(
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
    m.def("broadcast_mul", &broadcast_mul_forward, "Shared memory broadcast multiplication");
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
    if not A.is_cuda: 
        A = A.cuda()
    if not B.is_cuda: 
        B = B.cuda()
    
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
