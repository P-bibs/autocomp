# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_102701/code_12.py
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

# -------------------------------------------------------------------------
# Optimized CUDA kernel with register-efficient broadcast
# -------------------------------------------------------------------------
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
    // Each thread handles 4 elements (float4 vector)
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    const int total_elements = N * M;

    if (idx >= total_elements) return;

    // ---------- Fast path: 4 elements belong to the same row ----------
    // This is true for the overwhelming majority of threads.
    if (idx + 3 < total_elements) {
        int row0 = idx / M;
        int row3 = (idx + 3) / M;
        if (row0 == row3) {
            // Load A once into a register and broadcast
            float a = A[row0];

            // Load 4 consecutive B values as a float4
            float4 b_vec = reinterpret_cast<const float4*>(B)[idx / 4];

            // Element-wise multiply with the same 'a'
            float4 out_vec;
            out_vec.x = a * b_vec.x;
            out_vec.y = a * b_vec.y;
            out_vec.z = a * b_vec.z;
            out_vec.w = a * b_vec.w;

            // Store the result
            reinterpret_cast<float4*>(output)[idx / 4] = out_vec;
            return;
        }
    }

    // ---------- Fallback: handle row boundary or out-of-range ----------
    // This path is taken only for the few threads that cross a row boundary.
    for (int i = 0; i < 4 && idx + i < total_elements; ++i) {
        int cur = idx + i;
        int row = cur / M;
        output[cur] = A[row] * B[cur];
    }
}

void broadcast_mul_forward(const torch::Tensor& A,
                           const torch::Tensor& B,
                           torch::Tensor& output) {
    const int N = A.size(0);
    const int M = B.size(1);
    const int total_elements = N * M;

    const int threads_per_block = 256;
    const int blocks = (total_elements + 4 * threads_per_block - 1) /
                       (4 * threads_per_block);

    broadcast_mul_vectorized_kernel<<<blocks, threads_per_block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        output.data_ptr<float>(),
        N,
        M
    );
}
"""

# -------------------------------------------------------------------------
# C++ bindings (PyBind11)
# -------------------------------------------------------------------------
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

# -------------------------------------------------------------------------
# Build the inline CUDA extension
# -------------------------------------------------------------------------
fused_ext = load_inline(
    name='broadcast_mul_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# -------------------------------------------------------------------------
# Functional model exposed for evaluation
# -------------------------------------------------------------------------
def functional_model(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Perform A (Nx1) broadcast multiply with B (NxM) on the GPU."""
    if not A.is_cuda:
        A = A.cuda()
    if not B.is_cuda:
        B = B.cuda()

    output = torch.empty_like(B)          # same shape as B
    fused_ext.broadcast_mul(A, B, output) # launch the custom kernel
    return output

# -------------------------------------------------------------------------
# Helper functions required by the benchmark harness
# -------------------------------------------------------------------------
M = 4096
N = 4096

def get_init_inputs():
    """No special initialization needed for this benchmark."""
    return []

def get_inputs():
    """Return a fresh (A, B) pair for each measurement."""
    A = torch.rand(N)          # shape (N,)
    B = torch.rand(N, M)       # shape (N, M)
    return [A, B]
