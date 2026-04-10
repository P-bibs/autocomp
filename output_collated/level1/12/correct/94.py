# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_104927/code_13.py
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
# CUDA kernel: one block per row, vectorised float4 processing
# ----------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void broadcast_mul_row_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ output,
    int N,
    int M
) {
    // Row index comes from blockIdx (no division needed)
    const int row = blockIdx.x;
    if (row >= N) return;

    // Scalar broadcast value for this whole row
    const float a = A[row];

    const int tid = threadIdx.x;
    // Number of float4 segments in a row
    const int segments = (M + 3) / 4;

    // Each thread processes multiple segments in a strided way
    for (int seg = tid; seg < segments; seg += blockDim.x) {
        // Load 4 consecutive elements from B (vectorised)
        const float4 b_vec = reinterpret_cast<const float4*>(B + row * M)[seg];
        // Multiply by the broadcast scalar
        float4 out_vec;
        out_vec.x = a * b_vec.x;
        out_vec.y = a * b_vec.y;
        out_vec.z = a * b_vec.z;
        out_vec.w = a * b_vec.w;
        // Store the result vectorised
        reinterpret_cast<float4*>(output + row * M)[seg] = out_vec;
    }

    // Handle the tail when M is not a multiple of 4 (rare)
    if (tid == 0 && (M % 4 != 0)) {
        int seg = segments;               // first non‑full segment
        int base = seg * 4;
        for (int i = base; i < M; ++i) {
            output[row * M + i] = a * B[row * M + i];
        }
    }
}

// Host function that launches the kernel
void broadcast_mul_forward(const torch::Tensor& A,
                           const torch::Tensor& B,
                           torch::Tensor& output) {
    const int N = A.size(0);
    const int M = B.size(1);

    // One block per row; block size = 256 (multiple of 32) gives good occupancy
    const int threads_per_block = 256;
    const int blocks = N;   // N rows

    broadcast_mul_row_kernel<<<blocks, threads_per_block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        output.data_ptr<float>(),
        N,
        M
    );
}
"""

# ----------------------------------------------------------------------
# C++ binding (PYBIND11)
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>
void broadcast_mul_forward(const torch::Tensor& A,
                           const torch::Tensor& B,
                           torch::Tensor& output);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("broadcast_mul", &broadcast_mul_forward,
          "Vectorized broadcast multiplication");
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
# Public functional interface (must match the original signature)
# ----------------------------------------------------------------------
def functional_model(A, B):
    if not A.is_cuda:
        A = A.cuda()
    if not B.is_cuda:
        B = B.cuda()
    # Output has the same shape as B
    output = torch.empty_like(B)
    fused_ext.broadcast_mul(A, B, output)
    return output

# ----------------------------------------------------------------------
# Helper functions for the benchmark harness (optional)
# ----------------------------------------------------------------------
def get_init_inputs():
    return []   # No special initialisation needed

def get_inputs():
    N = 4096
    M = 4096
    A = torch.rand(N)       # broadcast vector
    B = torch.rand(N, M)    # matrix to be scaled
    return [A, B]
