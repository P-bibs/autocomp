# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_110950/code_9.py
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

template <typename scalar_t>
__global__ void broadcast_mul_shared_kernel(
    const scalar_t* __restrict__ A,          // [N]
    const scalar_t* __restrict__ B,          // [N, M]
    scalar_t* __restrict__ output,           // [N, M]
    int N,
    int M)
{
    // --------------------------------------------------------------
    // One block = one row (n).  The block may process several rows
    // if N > gridDim.x (grid‑stride over rows as well).
    // --------------------------------------------------------------
    for (int row = blockIdx.x; row < N; row += gridDim.x) {

        // ----------------------------------------------------------
        // Load the broadcast scalar into shared memory (single load)
        // ----------------------------------------------------------
        __shared__ scalar_t a_shared;
        if (threadIdx.x == 0) {
            a_shared = A[row];
        }
        __syncthreads();                       // broadcast to the whole block

        const scalar_t a_val = a_shared;        // register copy for the loop

        // ----------------------------------------------------------
        // Grid‑stride loop over the columns of this row.
        // Every thread writes to consecutive elements, so memory is
        // perfectly coalesced.
        // ----------------------------------------------------------
        for (int col = threadIdx.x; col < M; col += blockDim.x) {
            int idx = row * M + col;            // flat index
            output[idx] = a_val * B[idx];
        }
    }
}

// ------------------------------------------------------------------
// C‑wrapper called from Python (pybind11)
// ------------------------------------------------------------------
void broadcast_mul_forward(const torch::Tensor& A,
                           const torch::Tensor& B,
                           torch::Tensor& output)
{
    const int N = A.size(0);
    const int M = B.size(1);

    const int threads = 256;                     // multiple of 32 → good occupancy
    const int blocks  = std::min(N, 65535);      // RTX 2080 Ti: max gridDim.x = 2^31‑1,
                                                // but we keep it sane for very large N

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "broadcast_mul_forward", ([&] {
        broadcast_mul_shared_kernel<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            N,
            M);
    }));

    // Optional: check for launch errors (kept minimal for speed)
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("CUDA error in broadcast_mul_forward: ", cudaGetErrorString(err));
    }
}
"""

cpp_source = r"""
#include <torch/extension.h>

// Forward declaration of the CUDA kernel wrapper
void broadcast_mul_forward(const torch::Tensor& A,
                           const torch::Tensor& B,
                           torch::Tensor& output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("broadcast_mul", &broadcast_mul_forward,
          "Broadcast‑multiply A (N) with B (N×M) using shared memory");
}
"""

fused_ext = load_inline(
    name='broadcast_mul_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(A, B):
    if not A.is_cuda: A = A.cuda()
    if not B.is_cuda: B = B.cuda()
    
    output = torch.empty_like(B)
    fused_ext.broadcast_mul(A, B, output)
    return output

M = 4096
N = 4096

def get_init_inputs():
    return []  # No special initialization inputs needed

def get_inputs():
    A = torch.rand(N)
    B = torch.rand(N, M)
    return [A, B]

