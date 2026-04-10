# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_112116/code_11.py
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




# ==============================
# high_perf_broadcast_mul.py
# ==============================
import torch
from torch.utils.cpp_extension import load_inline

# ----------------------------------------------------------------------
# CUDA kernel – uses a grid‑stride loop (Optimization #7)
# ----------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void broadcast_mul_kernel(
    const scalar_t* __restrict__ A,   // (N,)
    const scalar_t* __restrict__ B,   // (N, M)
    scalar_t* __restrict__ C,         // (N, M) output
    const int64_t N,
    const int64_t M)
{
    // Global thread id
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Total number of threads in the grid
    const int64_t stride = gridDim.x * blockDim.x;

    // Process many elements per thread using a grid‑stride loop
    for (int64_t linear = idx; linear < N * M; linear += stride) {
        // Derive row (n) and column (m) from the linear index
        const int64_t n = linear / M;  // row index
        const int64_t m = linear % M;  // column index

        // Load once from A (broadcasted) and once from B
        const scalar_t a_val = A[n];
        const scalar_t b_val = B[linear];   // B is stored row‑major (contiguous in m)

        // Write the result
        C[linear] = a_val * b_val;
    }
}

// Dispatch helper – selects the correct scalar type
void broadcast_mul_forward(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C)
{
    const int64_t N = A.size(0);
    const int64_t M = B.size(1);

    const int threads = 256;
    const int blocks = (N * M + threads - 1) / threads;   // enough blocks for all elements

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(A.scalar_type(), "broadcast_mul_forward", ([&] {
        broadcast_mul_kernel<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            N,
            M);
    }));

    // Propagate possible launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("CUDA error in broadcast_mul_forward: ", cudaGetErrorString(err));
    }
}
"""

# ----------------------------------------------------------------------
# C++ binding (PyBind11) – required by the instructions
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

// Declaration of the CUDA launcher defined above
void broadcast_mul_forward(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C);

// PyBind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("broadcast_mul", &broadcast_mul_forward,
          "Broadcasted element‑wise multiplication (CUDA, grid‑stride loop)");
}
"""

# ----------------------------------------------------------------------
# Build the extension (the build flags already contain -O3 and --use_fast_math)
# ----------------------------------------------------------------------
fused_ext = load_inline(
    name='broadcast_mul_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True,
    verbose=False  # set True for debugging builds
)

# ----------------------------------------------------------------------
# Public API – functional_model
# ----------------------------------------------------------------------
def functional_model(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs  C = A.unsqueeze(1) * B   where
        A : (N,)
        B : (N, M)

    The implementation uses a single custom CUDA kernel that
    employs a grid‑stride loop (Optimization #7) to maximise GPU utilisation.
    """
    # ---- Input validation -------------------------------------------------
    if not (A.is_cuda and B.is_cuda):
        raise RuntimeError("Both inputs must be CUDA tensors")
    if A.dim() != 1:
        raise RuntimeError(f"A must be 1‑D, got shape {A.shape}")
    if B.dim() != 2:
        raise RuntimeError(f"B must be 2‑D, got shape {B.shape}")
    if A.size(0) != B.size(0):
        raise RuntimeError(f"First dimension mismatch: {A.size(0)} vs {B.size(0)}")

    # ---- Prepare output ----------------------------------------------------
    N, M = B.shape
    C = torch.empty_like(B)   # same dtype/device as B

    # ---- Call the custom kernel --------------------------------------------
    fused_ext.broadcast_mul(A, B, C)

    return C

# ----------------------------------------------------------------------
# Helper functions – kept for compatibility with the original script
# ----------------------------------------------------------------------
def get_init_inputs():
    # No special init required for the custom kernel
    return []

def get_inputs():
    # Use the same shapes as the original benchmark
    N = 4096
    M = 4096
    A = torch.rand(N, device='cuda', dtype=torch.float32)
    B = torch.rand(N, M, device='cuda', dtype=torch.float32)
    return [A, B]

# ----------------------------------------------------------------------
# Simple sanity‑check (can be removed in the evaluation harness)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    A, B = get_inputs()
    C_ref = A.unsqueeze(1) * B
    C_opt = functional_model(A, B)
    # Verify correctness (relative error < 1e-6)
    rel_err = (C_ref - C_opt).abs().max() / C_ref.abs().max()
    print(f"Max relative error: {rel_err.item():.2e}")
    # Quick timing benchmark
    import time
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(50):
        C_opt = functional_model(A, B)
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"Average latency (50 runs): {(t1-t0)/50*1000:.3f} ms")
