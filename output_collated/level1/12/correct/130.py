# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_105940/code_29.py
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

# ----------------------------------------------------------------------
# Optimized CUDA kernel
# ----------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void broadcast_mul_vectorized_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ output,
    int N, int M, int elements_per_row_block)
{
    // Each block processes 1024 elements (256 threads * 4 elements/thread)
    // We determine the row index based on the block index and how many blocks fit in a row.
    int row = blockIdx.x / elements_per_row_block;
    
    // Safety check for N
    if (row >= N) return;

    // Base column index for this thread
    int col_base = (blockIdx.x % elements_per_row_block) * 1024 + threadIdx.x * 4;

    // Load A[row] into a register once per thread
    float a_val = A[row];

    const float* B_row = B + row * M;
    float* out_row = output + row * M;

    // Vectorized path
    if (col_base + 3 < M) {
        float4 b_vec = reinterpret_cast<const float4*>(B_row + col_base)[0];
        float4 out_vec;
        out_vec.x = a_val * b_vec.x;
        out_vec.y = a_val * b_vec.y;
        out_vec.z = a_val * b_vec.z;
        out_vec.w = a_val * b_vec.w;
        reinterpret_cast<float4*>(out_row + col_base)[0] = out_vec;
    } else {
        // Scalar tail for remainder
        for (int i = 0; i < 4; ++i) {
            int col = col_base + i;
            if (col < M) {
                out_row[col] = a_val * B_row[col];
            }
        }
    }
}

void broadcast_mul_forward(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& output) {
    const int N = A.size(0);
    const int M = B.size(1);
    
    const int threads_per_block = 256;
    const int elements_per_thread = 4;
    const int elements_per_block = threads_per_block * elements_per_thread;
    
    // How many blocks are needed to cover one row (M)
    int elements_per_row_block = (M + elements_per_block - 1) / elements_per_block;
    if (elements_per_row_block == 0) elements_per_row_block = 1;

    // Total blocks needed
    int total_blocks = N * elements_per_row_block;

    broadcast_mul_vectorized_kernel<<<total_blocks, threads_per_block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        output.data_ptr<float>(),
        N, M, elements_per_row_block
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void broadcast_mul_forward(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& output);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("broadcast_mul", &broadcast_mul_forward, "Optimized broadcast multiplication");
}
"""

# Compile extension
fused_ext = load_inline(
    name='broadcast_mul_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(A, B):
    """
    Computes output[n, m] = A[n] * B[n, m]
    Ensures inputs are on GPU and invokes the optimized kernel.
    """
    if not A.is_cuda: A = A.cuda()
    if not B.is_cuda: B = B.cuda()
    
    output = torch.empty_like(B)
    fused_ext.broadcast_mul(A, B, output)
    return output
