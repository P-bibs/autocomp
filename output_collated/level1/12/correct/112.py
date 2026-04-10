# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_105940/code_5.py
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

#define ELEMENTS_PER_THREAD 4

__global__ void broadcast_mul_vectorized_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ output,
    int N,
    int M
) {
    // Process data row by row to maximize A reuse
    // Each thread block processes one row of B
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int threads_per_block = blockDim.x;
    
    if (row >= N) return;
    
    // Load A value for this row once and reuse
    float a_val = A[row];
    
    // Process elements in this row with vectorization
    int elements_per_thread = (M + threads_per_block * ELEMENTS_PER_THREAD - 1) / (threads_per_block * ELEMENTS_PER_THREAD);
    int start_idx = tid * ELEMENTS_PER_THREAD;
    
    const float4* b_row_ptr = reinterpret_cast<const float4*>(&B[row * M]);
    float4* out_row_ptr = reinterpret_cast<float4*>(&output[row * M]);
    
    for (int i = 0; i < elements_per_thread; i++) {
        int base_idx = start_idx + i * threads_per_block * ELEMENTS_PER_THREAD;
        
        if (base_idx + 3 < M) {
            // Safe to load full float4
            float4 b_vec = b_row_ptr[base_idx / 4];
            float4 out_vec;
            
            out_vec.x = a_val * b_vec.x;
            out_vec.y = a_val * b_vec.y;
            out_vec.z = a_val * b_vec.z;
            out_vec.w = a_val * b_vec.w;
            
            out_row_ptr[base_idx / 4] = out_vec;
        } else {
            // Handle remainder elements scalarly
            for (int j = 0; j < 4 && base_idx + j < M; j++) {
                int idx = base_idx + j;
                output[row * M + idx] = a_val * B[row * M + idx];
            }
        }
    }
}

void broadcast_mul_forward(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& output) {
    const int N = A.size(0);
    const int M = B.size(1);
    
    // Launch one block per row to maximize A reuse
    const int blocks = N;
    // Use enough threads to efficiently process each row
    const int threads_per_block = min(256, (M + ELEMENTS_PER_THREAD - 1) / ELEMENTS_PER_THREAD);
    
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
    m.def("broadcast_mul", &broadcast_mul_forward, "Optimized vectorized broadcast multiplication");
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
    # Ensure inputs are contiguous for proper memory access
    if not A.is_contiguous(): A = A.contiguous()
    if not B.is_contiguous(): B = B.contiguous()
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
