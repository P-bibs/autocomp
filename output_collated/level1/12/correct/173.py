# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_112116/code_14.py
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
# CUDA kernel – caches A[n] in shared memory to reduce global memory traffic
# -------------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// Kernel that caches the broadcast scalar in shared memory
__global__ void broadcast_mul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ output,
    int N,
    int M
) {
    // Shared memory to hold the scalar A[n] for the current block
    __shared__ float a_val_cache;

    int m_idx = blockIdx.x * (blockDim.x * 4) + threadIdx.x * 4;
    int n = blockIdx.y;

    // Load A[n] once per block into shared memory
    if (threadIdx.x == 0) {
        a_val_cache = A[n];
    }
    __syncthreads();

    // Perform the actual computation only when the thread handles a valid element
    if (n < N && m_idx < M) {
        float a_val = a_val_cache;               // use the cached value

        // Vectorized load-multiply-store (float4) for coalesced memory access
        float4 b_val = reinterpret_cast<const float4*>(&B[n * M + m_idx])[0];

        float4 out_val;
        out_val.x = a_val * b_val.x;
        out_val.y = a_val * b_val.y;
        out_val.z = a_val * b_val.z;
        out_val.w = a_val * b_val.w;

        reinterpret_cast<float4*>(&output[n * M + m_idx])[0] = out_val;
    }
}

// Host function that launches the kernel
void broadcast_mul_forward(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& output) {
    const int N = A.size(0);
    const int M = B.size(1);

    const int threads = 256;           // 256 threads per block
    const int elements_per_thread = 4; // each thread handles 4 floats

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

# -------------------------------------------------------------------------
# C++ binding (pybind11)
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>
void broadcast_mul_forward(const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& output);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("broadcast_mul", &broadcast_mul_forward,
          "Broadcast multiplication with shared-memory caching");
}
"""

# -------------------------------------------------------------------------
# Build the extension
# -------------------------------------------------------------------------
fused_ext = load_inline(
    name='broadcast_mul_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# -------------------------------------------------------------------------
# Functional model used for evaluation
# -------------------------------------------------------------------------
def functional_model(A, B):
    if not A.is_cuda:
        A = A.cuda()
    if not B.is_cuda:
        B = B.cuda()

    output = torch.empty_like(B)
    fused_ext.broadcast_mul(A, B, output)
    return output


# -------------------------------------------------------------------------
# Test parameters (not part of the final submission)
# -------------------------------------------------------------------------
N, M = 4096, 4096

def get_inputs():
    return [torch.rand(N).cuda(), torch.rand(N, M).cuda()]
