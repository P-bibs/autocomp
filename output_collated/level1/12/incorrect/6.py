# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_110950/code_13.py
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
# CUDA kernel – tiled, vectorised broadcast multiplication
# ------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void broadcast_mul_tile_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ output,
    int N,
    int M
) {
    // blockIdx.y -> row (no division needed)
    const int row = blockIdx.y;
    if (row >= N) return;

    // Load the broadcast value for this row once per block
    float a = A[row];

    // Each block handles a horizontal tile of width blockDim.x * 4 (= 512)
    const int tile_width = blockDim.x * 4;      // 512
    const int col_base   = blockIdx.x * tile_width;
    const int tid        = threadIdx.x;
    const int col        = col_base + tid * 4;   // first column this thread handles

    // Fully vectorised path: 4 consecutive elements fit into the row
    if (col + 3 < M) {
        const float4 b_vec = reinterpret_cast<const float4*>(B + row * M)[tid];
        float4 out_vec;
        out_vec.x = a * b_vec.x;
        out_vec.y = a * b_vec.y;
        out_vec.z = a * b_vec.z;
        out_vec.w = a * b_vec.w;
        reinterpret_cast<float4*>(output + row * M)[tid] = out_vec;
    } else {
        // Handle the remaining 0-3 elements scalar-wise
        for (int i = 0; i < 4 && col + i < M; ++i) {
            output[row * M + col + i] = a * B[row * M + col + i];
        }
    }
}

void broadcast_mul_forward(const torch::Tensor& A,
                           const torch::Tensor& B,
                           torch::Tensor& output) {
    const int N = A.size(0);
    const int M = B.size(1);

    const int threads_per_block = 128;                 // 128 threads → 512 floats per block
    const int tile_width = threads_per_block * 4;      // 512
    const int blocks_x   = (M + tile_width - 1) / tile_width;
    const int blocks_y   = N;

    dim3 blocks(blocks_x, blocks_y);
    broadcast_mul_tile_kernel<<<blocks, threads_per_block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        output.data_ptr<float>(),
        N,
        M
    );
}
"""

# ------------------------------------------------------------
# C++ binding (pybind11)
# ------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>
void broadcast_mul_forward(const torch::Tensor& A,
                           const torch::Tensor& B,
                           torch::Tensor& output);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("broadcast_mul", &broadcast_mul_forward,
          "Vectorised broadcast multiplication with tiling");
}
"""

# ------------------------------------------------------------
# Build the inline extension
# ------------------------------------------------------------
fused_ext = load_inline(
    name='broadcast_mul_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# ------------------------------------------------------------
# Functional interface used by the evaluator
# ------------------------------------------------------------
def functional_model(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs A (shape [N]) broadcast-multiplied with B (shape [N, M]).
    Both inputs must be on the GPU; the result has the same shape as B.
    """
    if not A.is_cuda:
        A = A.cuda()
    if not B.is_cuda:
        B = B.cuda()

    output = torch.empty_like(B)          # same device, dtype, shape as B
    fused_ext.broadcast_mul(A, B, output)
    return output


# ------------------------------------------------------------
# Helper functions required by the benchmarking harness
# ------------------------------------------------------------
N = 4096
M = 4096

def get_init_inputs():
    """No special initialisation required."""
    return []

def get_inputs():
    """Return a fresh pair of random tensors."""
    A = torch.rand(N)          # vector of length N
    B = torch.rand(N, M)       # N×M matrix
    return [A, B]
