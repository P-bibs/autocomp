# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_103853/code_29.py
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

# ------------------------------------------------------------
# CUDA kernel – row-parallel, shared-memory cached A
# ------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void broadcast_mul_row_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ output,
    const int N,
    const int M
) {
    // Each block handles one row
    const int row = blockIdx.x;
    if (row >= N) return;

    // Cache A[row] in shared memory
    __shared__ float a_val;
    if (threadIdx.x == 0) {
        a_val = A[row];
    }
    __syncthreads();
    
    const float a = a_val;
    const float* B_row = B + (size_t)row * M;
    float* out_row = output + (size_t)row * M;

    // Process row in chunks of 4 (float4)
    // Coalesced access pattern: threads in a warp access consecutive B elements
    int col_step = blockDim.x * 4;
    for (int col = threadIdx.x * 4; col < M; col += col_step) {
        if (col + 3 < M) {
            float4 b_vec = reinterpret_cast<const float4*>(B_row + col)[0];
            float4 out_vec;
            out_vec.x = a * b_vec.x;
            out_vec.y = a * b_vec.y;
            out_vec.z = a * b_vec.z;
            out_vec.w = a * b_vec.w;
            reinterpret_cast<float4*>(out_row + col)[0] = out_vec;
        } else {
            // Handle remaining elements if M is not a multiple of 4
            for (int i = col; i < M && i < col + 4; ++i) {
                out_row[i] = a * B_row[i];
            }
        }
    }
}

void broadcast_mul_forward(const torch::Tensor& A,
                           const torch::Tensor& B,
                           torch::Tensor& output) {
    const int N = A.size(0);
    const int M = B.size(1);
    
    const int threads = 256;
    const int blocks = N; 

    broadcast_mul_row_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        output.data_ptr<float>(),
        N,
        M
    );
}
"""

# ------------------------------------------------------------
# C++ binding (PyBind11)
# ------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void broadcast_mul_forward(const torch::Tensor& A,
                           const torch::Tensor& B,
                           torch::Tensor& output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("broadcast_mul", &broadcast_mul_forward, "Vectorized broadcast multiplication");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='broadcast_mul_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(A, B):
    """
    Highly optimized broadcast multiplication: out[i, j] = A[i] * B[i, j].
    Uses a row-parallel kernel with A cached in shared memory.
    """
    if not A.is_cuda: A = A.cuda()
    if not B.is_cuda: B = B.cuda()
    
    # Ensure inputs are contiguous for row-based indexing
    if not A.is_contiguous(): A = A.contiguous()
    if not B.is_contiguous(): B = B.contiguous()
    
    output = torch.empty_like(B)
    fused_ext.broadcast_mul(A, B, output)
    return output

def get_init_inputs():
    return []

def get_inputs():
    N, M = 4096, 4096
    A = torch.rand(N)
    B = torch.rand(N, M)
    return [A, B]
