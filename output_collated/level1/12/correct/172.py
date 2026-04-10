# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_112116/code_13.py
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

// Kernel: broadcast multiply A (N,) with B (N,M) -> output (N,M)
//   output[i,j] = A[i] * B[i,j]
//
// Each thread handles 4 output elements (float4).
// To minimize random global reads of A, we cache the needed A values
// in shared memory.  A block processes at most two rows, so we store
// up to two A values.

__global__ void broadcast_mul_vectorized_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ output,
    int N,
    int M
) {
    // Shared memory for caching A values needed by this block.
    // Maximum number of rows per block is 2 (see analysis).
    __shared__ float A_cache[2];

    // Global element index where this block starts.
    const int block_start_idx = blockIdx.x * blockDim.x * 4;

    // Row index of the first and last element processed by this block.
    const int row_start = block_start_idx / M;
    const int row_end   = min((block_start_idx + blockDim.x * 4 - 1) / M, N - 1);

    // Load the needed A values into shared memory.
    // Thread 0 loads A[row_start]; thread 1 loads A[row_end] if needed.
    if (threadIdx.x == 0) {
        A_cache[0] = A[row_start];
    }
    if (threadIdx.x == 1 && row_end > row_start) {
        A_cache[1] = A[row_end];
    }
    // Wait for the cache to be filled before any thread uses it.
    __syncthreads();

    // Compute the global element index for this thread.
    const int idx = block_start_idx + threadIdx.x * 4;
    const int total_elements = N * M;

    // Process a full vector of 4 elements if possible.
    if (idx + 3 < total_elements) {
        // Coalesced float4 load from B.
        float4 b_vec = reinterpret_cast<const float4*>(B)[idx / 4];

        // Row indices for the 4 elements.
        const int n0 = (idx)       / M;
        const int n1 = (idx + 1)   / M;
        const int n2 = (idx + 2)   / M;
        const int n3 = (idx + 3)   / M;

        // Retrieve the cached A values.
        const float a0 = A_cache[n0 - row_start];
        const float a1 = A_cache[n1 - row_start];
        const float a2 = A_cache[n2 - row_start];
        const float a3 = A_cache[n3 - row_start];

        // Element-wise multiplication.
        float4 out_vec;
        out_vec.x = a0 * b_vec.x;
        out_vec.y = a1 * b_vec.y;
        out_vec.z = a2 * b_vec.z;
        out_vec.w = a3 * b_vec.w;

        // Coalesced float4 store to output.
        reinterpret_cast<float4*>(output)[idx / 4] = out_vec;
    } else {
        // Handle remaining (0-3) elements scalar-wise.
        #pragma unroll
        for (int i = 0; i < 4 && idx + i < total_elements; ++i) {
            const int current_idx = idx + i;
            const int n = current_idx / M;
            if (n < N) { // Boundary check for A access
                const float a = A_cache[n - row_start];
                output[current_idx] = a * B[current_idx];
            }
        }
    }
}

// Host function that launches the kernel with appropriate grid/block size.
void broadcast_mul_forward(const torch::Tensor& A,
                           const torch::Tensor& B,
                           torch::Tensor& output) {
    const int N = A.size(0);
    const int M = B.size(1);
    const int total_elements = N * M;

    const int threads_per_block = 256;
    // Each thread processes 4 elements.
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

# C++ interface exposed via pybind11
cpp_source = r"""
#include <torch/extension.h>

void broadcast_mul_forward(const torch::Tensor& A,
                           const torch::Tensor& B,
                           torch::Tensor& output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("broadcast_mul", &broadcast_mul_forward,
          "Vectorized broadcast multiplication (A * B) with shared-memory caching");
}
"""

# Build the inline CUDA extension.
fused_ext = load_inline(
    name='broadcast_mul_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# --------------------------------------------------------------------
# Functional wrapper that will be imported / evaluated.
# --------------------------------------------------------------------
def functional_model(A, B):
    """
    Compute output = A (size N) broadcast-multiplied with B (size N×M).
    A and B are moved to GPU if not already there.
    """
    if not A.is_cuda:
        A = A.cuda()
    if not B.is_cuda:
        B = B.cuda()

    # Allocate output tensor with same shape as B.
    output = torch.empty_like(B)
    fused_ext.broadcast_mul(A, B, output)
    return output
