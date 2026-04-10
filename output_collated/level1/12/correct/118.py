# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_105940/code_13.py
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
# CUDA kernel (register‑friendly, row‑level indexing)
# ----------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void broadcast_mul_vectorized_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ output,
    int N, int M, int blocks_per_row)
{
    // ----- row and column base for this block/thread -----
    int row = blockIdx.x / blocks_per_row;                       // one row per block
    int col_base = (blockIdx.x % blocks_per_row) * (blockDim.x * 4) + threadIdx.x * 4;

    // ----- load A[row] once per thread (register) -----
    float a_val = A[row];

    // ----- pointers to the current row of B and output -----
    const float* B_row = B + row * M;
    float* out_row = output + row * M;

    // ----- vectorised path: 4 consecutive elements -----
    if (col_base + 3 < M) {
        float4 b_vec = reinterpret_cast<const float4*>(B_row + col_base)[0];
        float4 out_vec;
        out_vec.x = a_val * b_vec.x;
        out_vec.y = a_val * b_vec.y;
        out_vec.z = a_val * b_vec.z;
        out_vec.w = a_val * b_vec.w;
        reinterpret_cast<float4*>(out_row + col_base)[0] = out_vec;
    } else {
        // ----- scalar tail (up to 3 elements) -----
        for (int i = 0; i < 4 && col_base + i < M; ++i) {
            out_row[col_base + i] = a_val * B_row[col_base + i];
        }
    }
}

void broadcast_mul_forward(const torch::Tensor& A,
                           const torch::Tensor& B,
                           torch::Tensor& output) {
    const int N = A.size(0);
    const int M = B.size(1);
    const int total_elements = N * M;

    const int threads_per_block = 256;
    const int elements_per_thread = 4;               // each thread handles 4 floats
    const int block_size = threads_per_block * elements_per_thread; // 1024

    // Number of blocks that cover one complete row
    const int blocks_per_row = (M + block_size - 1) / block_size;       // 4 for M=4096
    const int total_blocks = (total_elements + block_size - 1) / block_size;

    broadcast_mul_vectorized_kernel<<<total_blocks, threads_per_block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        output.data_ptr<float>(),
        N, M, blocks_per_row
    );
}
"""

# ----------------------------------------------------------------------
# C++ interface (pybind11)
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>
void broadcast_mul_forward(const torch::Tensor& A,
                           const torch::Tensor& B,
                           torch::Tensor& output);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("broadcast_mul", &broadcast_mul_forward,
          "Vectorized broadcast multiplication (A[:,None] * B)");
}
"""

# ----------------------------------------------------------------------
# Build the inline extension
# ----------------------------------------------------------------------
fused_ext = load_inline(
    name='broadcast_mul_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# ----------------------------------------------------------------------
# Functional wrapper that will be imported
# ----------------------------------------------------------------------
def functional_model(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Computes output[n, m] = A[n] * B[n, m] using a custom CUDA kernel.
    Both inputs are moved to GPU if they are not already there.
    """
    if not A.is_cuda:
        A = A.cuda()
    if not B.is_cuda:
        B = B.cuda()

    output = torch.empty_like(B)          # same shape & layout as B
    fused_ext.broadcast_mul(A, B, output)  # launch the optimized kernel
    return output

M = 4096
N = 4096

def get_init_inputs():
    return []  # No special initialization inputs needed

def get_inputs():
    A = torch.rand(N)
    B = torch.rand(N, M)
    return [A, B]
