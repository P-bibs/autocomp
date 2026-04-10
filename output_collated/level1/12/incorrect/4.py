# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_105940/code_8.py
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
# CUDA kernel – now uses shared memory to cache A
# -------------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#ifndef __CUDA_ARCH__
#define __CUDA_ARCH__ 0
#endif

// ---------------------------------------------------------------
//  Vectorised kernel
// ---------------------------------------------------------------
extern "C" __global__
void fused_op_forward_kernel(
    const float* __restrict__ A,          // [N]
    const float* __restrict__ B,          // [N, M] row‑major
    float*       __restrict__ C,          // [N, M] row‑major
    int N,
    int M
) {
    // -----------------------------------------------------------
    // 1) shared‑memory tile for one block of A
    // -----------------------------------------------------------
    extern __shared__ float shA[];               // size = blockDim.x * sizeof(float)

    // Global row handled by this thread (one row per thread)
    int global_row = blockIdx.x * blockDim.x + threadIdx.x;

    // -----------------------------------------------------------
    // 2) Load our A[row] into shared memory (once per thread)
    // -----------------------------------------------------------
    if (global_row < N) {
        shA[threadIdx.x] = A[global_row];
    }
    // Everyone can safely read its own entry now
    __syncthreads();

    // -----------------------------------------------------------
    // 3) Vectorised processing of B / C (4 floats per iteration)
    // -----------------------------------------------------------
    // Each thread will walk over its row in steps of (blockDim.x * gridDim.x * 4)
    //   idx is the *float* index (not element‑wise)
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;   // start index for this thread
    int total_elements = N * M;

    for (int i = idx; i < total_elements; i += blockDim.x * gridDim.x * 4) {
        // Row belonging to element i
        int row = i / M;
        // Guard against out‑of‑range rows (possible for the last partial block)
        if (row >= N) break;

        // a_val comes from shared memory – no extra global load
        float a_val = shA[threadIdx.x];

        // -----------------------------------------------------------------
        // The next part is identical to the original implementation.
        // -----------------------------------------------------------------
        // If the 4‑wide vector is completely inside the current row we can
        // safely use float4 loads/stores.
        if (i + 3 < (row + 1) * M) {
            // reinterpret as float4 (always aligned because we step by 4)
            float4 b_vec = reinterpret_cast<const float4*>(&B[i])[0];
            float4 c_vec;
            c_vec.x = a_val * b_vec.x;
            c_vec.y = a_val * b_vec.y;
            c_vec.z = a_val * b_vec.z;
            c_vec.w = a_val * b_vec.w;
            reinterpret_cast<float4*>(&C[i])[0] = c_vec;
        } else {
            // Tail handling – at most 3 elements left in the row
            for (int j = 0; j < 4; ++j) {
                int flat = i + j;
                if (flat < total_elements && flat / M == row) {
                    C[flat] = a_val * B[flat];
                }
            }
        }
    }
}

// ---------------------------------------------------------------
//  Host‑side launcher (still a plain C++ function)
// ---------------------------------------------------------------
void fused_op_forward(const torch::Tensor& A,
                      const torch::Tensor& B,
                      torch::Tensor& C,
                      int N,
                      int M) {
    const int threads = 256;                     // one thread per row inside a block
    // Number of blocks = ceil(N / threads)
    int num_blocks = (N + threads - 1) / threads;
    // Limit to the maximum grid size supported on RTX2080Ti
    num_blocks = min(num_blocks, 65535);

    // Dynamic shared memory = threads * sizeof(float)
    size_t shmem_bytes = threads * sizeof(float);

    fused_op_forward_kernel<<<num_blocks, threads, shmem_bytes>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N,
        M
    );
}
"""

# -------------------------------------------------------------------------
# C++ binding – unchanged apart from the new kernel name
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

// Forward declaration
void fused_op_forward(const torch::Tensor& A,
                      const torch::Tensor& B,
                      torch::Tensor& C,
                      int N,
                      int M);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward,
          "Vector‑multiply B by per‑row scalar A (shared‑memory version)");
}
"""

# -------------------------------------------------------------------------
# Build the extension
# -------------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True,
    verbose=False
)

# -------------------------------------------------------------------------
# Public API – exactly the same signature as before
# -------------------------------------------------------------------------
def functional_model(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Multiply each row of B by the corresponding scalar in A.

    Args:
        A (torch.Tensor):   shape (N,)   – 1‑D tensor (float32, CUDA)
        B (torch.Tensor):   shape (N, M) – 2‑D tensor (float32, CUDA)

    Returns:
        torch.Tensor: C = A.unsqueeze(1) * B  (shape N×M, float32, CUDA)
    """
    # Ensure contiguous on the device (the kernel expects contiguous input)
    A_contig = A.contiguous().cuda()
    B_contig = B.contiguous().cuda()
    N, M = B_contig.shape

    # Allocate output
    C = torch.empty(N, M, dtype=torch.float32, device='cuda')

    # Call the fused kernel
    fused_ext.fused_op(A_contig, B_contig, C, N, M)

    return C
