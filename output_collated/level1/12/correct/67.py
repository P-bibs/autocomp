# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_103853/code_16.py
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

# Optimized CUDA Kernel
# 1. Replaced expensive `i / M` division with a 2D grid approach.
# 2. Maintained requested #pragma unroll for tail-end processing.
# 3. Used __restrict__ pointers for better cache utilization.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector_types.h>

__global__ void fused_op_forward_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N,
    int M
) {
    int row = blockIdx.y;
    if (row >= N) return;

    const float* A_row = A + row;
    const float* B_row = B + row * M;
    float* C_row = C + row * M;
    float a_val = *A_row;

    // Process vectorized elements
    int col_limit = M - (M % 4);
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = col * 4;

    if (idx < col_limit) {
        float4 b_vec = reinterpret_cast<const float4*>(&B_row[idx])[0];
        float4 c_vec;
        c_vec.x = a_val * b_vec.x;
        c_vec.y = a_val * b_vec.y;
        c_vec.z = a_val * b_vec.z;
        c_vec.w = a_val * b_vec.w;
        reinterpret_cast<float4*>(&C_row[idx])[0] = c_vec;
    } 
    
    // Tail handling using unrolled loop
    if (M % 4 != 0 && idx >= col_limit && idx < M) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int current_col = idx + j;
            if (current_col < M) {
                C_row[current_col] = a_val * B_row[current_col];
            }
        }
    }
}

void fused_op_forward(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& C, int N, int M) {
    // 1D grid for cols, 1D grid for rows
    int threads_x = 256;
    dim3 grid((M + (threads_x * 4) - 1) / (threads_x * 4), N);
    
    fused_op_forward_kernel<<<grid, threads_x>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N,
        M
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_op_forward(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& C, int N, int M);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Vectorized fused unsqueeze-multiply with unrolled tail");
}
"""

fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(A, B):
    N, M = B.shape
    C = torch.empty_like(B)
    # Ensure inputs are contiguous to satisfy float4 pointer alignment
    fused_ext.fused_op(A.contiguous(), B.contiguous(), C, N, M)
    return C
