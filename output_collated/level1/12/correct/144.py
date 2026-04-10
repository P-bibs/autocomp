# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_110950/code_14.py
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
# Optimized CUDA kernel using shared memory to reduce redundant reads
# ----------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// Vectorized kernel for broadcast multiplication.
// The scalar multiplier A[n] is loaded once per block and broadcast through
// shared memory, eliminating redundant global reads.
__global__ void broadcast_mul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ output,
    int N,
    int M
)
{
    // Shared memory for the broadcast scalar of the current row
    __shared__ float a_val;

    int n = blockIdx.y;                       // row index
    int m_idx = blockIdx.x * (blockDim.x * 4) + threadIdx.x * 4; // start element for this thread

    // ------------------------------------------------------------------
    // Load the scalar A[n] once per block (thread 0 does the fetch)
    // ------------------------------------------------------------------
    if (threadIdx.x == 0) {
        a_val = (n < N) ? A[n] : 0.0f;
    }
    __syncthreads();

    // ------------------------------------------------------------------
    // Main computation – each thread handles 4 consecutive floats (float4)
    // ------------------------------------------------------------------
    if (n < N && m_idx < M) {
        float a = a_val;                     // broadcasted multiplier

        // Coalesced read of B row
        const float* B_row = B + n * M;
        float4 b = reinterpret_cast<const float4*>(B_row + m_idx)[0];

        // Element-wise multiply
        float4 out;
        out.x = a * b.x;
        out.y = a * b.y;
        out.z = a * b.z;
        out.w = a * b.w;

        // Coalesced write to output
        float* out_row = output + n * M;
        reinterpret_cast<float4*>(out_row + m_idx)[0] = out;
    }
}

// Host-side launcher
void broadcast_mul_forward(const torch::Tensor& A,
                           const torch::Tensor& B,
                           torch::Tensor& output)
{
    const int N = A.size(0);
    const int M = B.size(1);

    // 256 threads -> 64 float4 elements per block
    const int threads = 256;
    const int elements_per_thread = 4;

    // Grid: (M/4)/threads in X, N in Y
    dim3 grid((M / elements_per_thread + threads - 1) / threads, N);

    broadcast_mul_kernel<<<grid, threads>>>(
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
          "Vectorized broadcast multiplication with shared-memory optimisation");
}
"""

# ----------------------------------------------------------------------
# Compile the inline CUDA extension
# ----------------------------------------------------------------------
fused_ext = load_inline(
    name='broadcast_mul_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# ----------------------------------------------------------------------
# Functional wrapper – same interface as the original functional_model
# ----------------------------------------------------------------------
def functional_model(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs  A (broadcast) * B  where:
        A : (N,)   vector
        B : (N, M) matrix
    Returns a tensor of shape (N, M).
    """
    # Ensure inputs are on CUDA
    if not A.is_cuda:
        A = A.cuda()
    if not B.is_cuda:
        B = B.cuda()

    # Allocate output (row-major, same layout as B)
    output = torch.empty_like(B)

    # Call the custom CUDA kernel
    fused_ext.broadcast_mul(A, B, output)

    return output


# ----------------------------------------------------------------------
# Helper to generate test inputs (used only for debugging/validation)
# ----------------------------------------------------------------------
def get_inputs():
    N = 4096
    M = 4096
    return [torch.rand(N).cuda(), torch.rand(N, M).cuda()]


# ----------------------------------------------------------------------
# Simple correctness check (optional, can be run manually)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    A, B = get_inputs()
    out = functional_model(A, B)
    # Compare with pure PyTorch reference (slow but correct)
    ref = A.unsqueeze(1) * B
    print("Max absolute difference:", (out - ref).abs().max().item())
