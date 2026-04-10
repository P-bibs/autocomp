# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_105940/code_11.py
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




# --------------------------------------------------------------
#  tiled_broadcast_mul.py
# --------------------------------------------------------------
import torch
from torch.utils.cpp_extension import load_inline

# --------------------------------------------------------------
#  CUDA kernel (tiling optimisation)
# --------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#ifndef TILE_M
#define TILE_M 32          // tile width  (columns)
#endif
#ifndef TILE_N
#define TILE_N 8           // tile height (rows)
#endif

// ----------------------------------------------------------------
//  Kernel: C = A.unsqueeze(1) * B
//  A : [N]                 (float)
//  B : [N, M]              (float, row‑major)
//  C : [N, M]              (float, row‑major)
// ----------------------------------------------------------------
template <typename scalar_t>
__global__ void broadcast_mul_tiled_kernel(
    const scalar_t* __restrict__ A,          // [N]
    const scalar_t* __restrict__ B,          // [N, M]
    scalar_t* __restrict__ C,                // [N, M]
    const int N,
    const int M)
{
    // ------------------------------------------------------------
    //  Tile dimensions (shared memory)
    // ------------------------------------------------------------
    __shared__ scalar_t tile[TILE_N][TILE_M + 1];   // +1 avoids bank conflict

    // ------------------------------------------------------------
    //  Row of A that this block works on.
    //  Each block processes TILE_N consecutive rows.
    // ------------------------------------------------------------
    int row_base = blockIdx.y * TILE_N;
    int col_base = blockIdx.x * TILE_M;

    // ------------------------------------------------------------
    //  Each thread loads one element of B (if inside the matrix)
    // ------------------------------------------------------------
    int thread_row = threadIdx.y;   // 0 .. TILE_N-1
    int thread_col = threadIdx.x;   // 0 .. TILE_M-1

    // Global indices for the element this thread will load
    int g_row = row_base + thread_row;
    int g_col = col_base + thread_col;

    // ------------------------------------------------------------
    //  Load B into shared memory (if inside bounds)
    // ------------------------------------------------------------
    if (g_row < N && g_col < M) {
        tile[thread_row][thread_col] = B[g_row * M + g_col];
    } else {
        // pad out‑of‑bounds with zero (won’t affect result)
        tile[thread_row][thread_col] = static_cast<scalar_t>(0);
    }

    __syncthreads();   // make sure the whole tile is resident

    // ------------------------------------------------------------
    //  Multiply by the broadcasted scalar A[g_row] and write C
    // ------------------------------------------------------------
    if (g_row < N && g_col < M) {
        scalar_t a_val = A[g_row];                     // broadcast scalar
        C[g_row * M + g_col] = a_val * tile[thread_row][thread_col];
    }
}

// ----------------------------------------------------------------
//  Launcher (templated for float/double)
//-----------------------------------------------------------------
void broadcast_mul_tiled_launcher(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C)
{
    const int N = A.size(0);
    const int M = B.size(1);

    const dim3 block(TILE_M, TILE_N);                     // 32×8 = 256 threads
    const dim3 grid(
        (M + TILE_M - 1) / TILE_M,
        (N + TILE_N - 1) / TILE_N
    );

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "broadcast_mul_tiled", ([&] {
        broadcast_mul_tiled_kernel<scalar_t><<<grid, block>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            N,
            M
        );
    }));
    // CUDA error check (optional but helpful)
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("CUDA error in broadcast_mul_tiled_launcher: ", cudaGetErrorString(err));
    }
}
"""

# --------------------------------------------------------------
#  C++/PyBind wrapper (required by load_inline)
# --------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

// Forward declaration of the launcher defined in the .cu file
void broadcast_mul_tiled_launcher(torch::Tensor A,
                                  torch::Tensor B,
                                  torch::Tensor C);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("broadcast_mul", &broadcast_mul_tiled_launcher,
          "Broadcast‑multiply A (N) with B (N×M) using tiled CUDA kernel");
}
"""

# --------------------------------------------------------------
#  Build the extension (once)
# --------------------------------------------------------------
fused_ext = load_inline(
    name='broadcast_mul_tiled',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cflags=['-O3'],
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True,
    verbose=False
)

# --------------------------------------------------------------
#  Optimised functional_model ------------------------------------------------
# --------------------------------------------------------------
def functional_model(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Computes   C = A.unsqueeze(1) * B
    using the tiled CUDA kernel defined above.
    A : (N,)      float32
    B : (N, M)    float32
    Returns C : (N, M)
    """
    assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors"
    assert A.dtype == torch.float32 and B.dtype == torch.float32, "Only float32 supported"
    N, M = B.shape
    assert A.shape == (N,)

    # allocate output tensor (same layout / device as B)
    C = torch.empty_like(B)

    # launch kernel
    fused_ext.broadcast_mul(A, B, C)

    return C

# --------------------------------------------------------------
#  Helpers (unchanged from the original script)
# --------------------------------------------------------------
M = 4096
N = 4096

def get_init_inputs():
    """No special initialization inputs needed."""
    return []  # placeholder – kept for compatibility with the harness

def get_inputs():
    """Generate random inputs on the GPU."""
    A = torch.rand(N, device='cuda', dtype=torch.float32)
    B = torch.rand(N, M, device='cuda', dtype=torch.float32)
    return [A, B]

# --------------------------------------------------------------
#  Quick sanity‑check (can be removed in production)
# --------------------------------------------------------------
if __name__ == "__main__":
    A, B = get_inputs()
    C_opt = functional_model(A, B)
    # Reference implementation using PyTorch (for correctness)
    C_ref = A.view(N, 1) * B
    torch.testing.assert_allclose(C_opt, C_ref, rtol=1e-5, atol=1e-6)
    print("Correctness check passed.")
