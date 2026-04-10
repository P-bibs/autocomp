# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_110950/code_10.py
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

# ------------------------------------------------------------------
#  CUDA kernel – loads a tile of B into shared memory, multiplies
#  with the broadcasted scalar A[row] and writes the result.
# ------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#ifndef BLOCK_THREADS
#define BLOCK_THREADS 64          // 64 threads -> 256 float elements per tile
#endif

// each thread processes 4 consecutive floats (float4)
#define ELEMENTS_PER_THREAD 4

// ------------------------------------------------------------------
//  Kernel
// ------------------------------------------------------------------
__global__ void broadcast_mul_shared_kernel(
    const float* __restrict__ A,          // (N,)
    const float* __restrict__ B,          // (N, M)
    float*       __restrict__ out,        // (N, M)
    const int N,
    const int M)
{
    // --------------------------------------------------------------
    //  Tile indices
    // --------------------------------------------------------------
    const int row        = blockIdx.y;                     // which row of A / B
    const int tile_start = blockIdx.x * BLOCK_THREADS * ELEMENTS_PER_THREAD;
    const int tx         = threadIdx.x;                    // 0 .. BLOCK_THREADS-1

    if (row >= N) return;                                 // safety

    // ----------------------------------------------------------------
    //  Load the scalar a = A[row] – broadcast to the whole block
    // ----------------------------------------------------------------
    const float a_val = A[row];

    // --------------------------------------------------------------
    //  Shared memory for one tile of B (256 floats = 1 KiB)
    // --------------------------------------------------------------
    __shared__ float shB[BLOCK_THREADS * ELEMENTS_PER_THREAD];

    // --------------------------------------------------------------
    //  Loop over column tiles (each tile = BLOCK_THREADS * 4 elements)
    // --------------------------------------------------------------
    for (int col = tile_start;
         col < M;
         col += gridDim.x * BLOCK_THREADS * ELEMENTS_PER_THREAD)
    {
        // ----------------------------------------------------------
        //  1) Load a float4 from global memory into shared memory
        // ----------------------------------------------------------
        int global_idx = row * M + col + tx * ELEMENTS_PER_THREAD;

        // Guard against the tail of the matrix
        if (col + tx * ELEMENTS_PER_THREAD + 3 < M) {
            // fully aligned load
            float4 b_vec = reinterpret_cast<const float4*>(B + global_idx)[0];
            // store the 4 floats into shared memory as a float4
            reinterpret_cast<float4*>(shB + tx * ELEMENTS_PER_THREAD)[0] = b_vec;
        } else {
            // tail handling – scalar loads
            #pragma unroll
            for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
                int idx = global_idx + i;
                shB[tx * ELEMENTS_PER_THREAD + i] =
                    (idx < row * M + M) ? B[idx] : 0.0f;   // out‑of‑bounds = 0
            }
        }

        __syncthreads();   // make sure the whole tile is in shared memory

        // ----------------------------------------------------------
        //  2) Multiply with the broadcast scalar and write back
        // ----------------------------------------------------------
        #pragma unroll
        for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
            int out_idx = global_idx + i;
            if (out_idx < row * M + M) {                // bounds check
                out[out_idx] = a_val * shB[tx * ELEMENTS_PER_THREAD + i];
            }
        }

        __syncthreads();   // before loading the next tile
    }
}

// ------------------------------------------------------------------
//  C++ wrapper (called from Python)
// ------------------------------------------------------------------
void broadcast_mul_forward(const torch::Tensor& A,
                           const torch::Tensor& B,
                           torch::Tensor& out)
{
    const int N = A.size(0);
    const int M = B.size(1);

    // 256 threads ( = BLOCK_THREADS ) per block
    const dim3 block(BLOCK_THREADS);
    const dim3 grid( (M + BLOCK_THREADS * ELEMENTS_PER_THREAD - 1)
                     / (BLOCK_THREADS * ELEMENTS_PER_THREAD), N );

    broadcast_mul_shared_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        out.data_ptr<float>(),
        N,
        M);
}
"""

# ------------------------------------------------------------------
#  Minimal C++/pybind11 glue – required by load_inline
# ------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

// Forward declaration of the kernel launcher defined in the .cu file
void broadcast_mul_forward(const torch::Tensor& A,
                           const torch::Tensor& B,
                           torch::Tensor& out);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("broadcast_mul", &broadcast_mul_forward,
          "Broadcast‑multiply A (N,) with B (N,M) using shared memory");
}
"""

# ------------------------------------------------------------------
#  Compile the extension
# ------------------------------------------------------------------
fused_ext = load_inline(
    name='broadcast_mul_shared_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# ------------------------------------------------------------------
#  Functional model – unchanged API
# ------------------------------------------------------------------
def functional_model(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Compute  out[i, j] = A[i] * B[i, j]  using the fused CUDA kernel.
    """
    if not A.is_cuda:
        A = A.cuda()
    if not B.is_cuda:
        B = B.cuda()

    out = torch.empty_like(B)
    fused_ext.broadcast_mul(A, B, out)
    return out

# ------------------------------------------------------------------
#  Helper to generate the required inputs (kept identical to the
#  original benchmark)
# ------------------------------------------------------------------
def get_inputs():
    N, M = 4096, 4096
    return [torch.rand(N, device='cuda'), torch.rand(N, M, device='cuda')]

