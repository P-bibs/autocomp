# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260405_222929/code_1.py
# fused_model_vectorized.py
# -------------------------------------------------------------
# Optimized ModelNew – vectorized (float4) reduction kernel
# -------------------------------------------------------------
# This file contains a single‑file implementation of the original
# ModelNew but replaces the per‑element reduction with a vectorised
# load (float4) to improve memory‑bandwidth utilisation on RTX 2080 Ti.
# All PyTorch operations are fused into one CUDA kernel; the API
# remains identical to the reference implementation.
# -------------------------------------------------------------
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ------------------------------------------------------------------
# CUDA kernel source (inline) – vectorised reduction with float4 loads
# ------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <type_traits>

template <typename scalar_t>
__device__ __forceinline__ scalar_t warp_reduce_sum(scalar_t val) {
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

/*
 * One block processes a single row of shape (out_features,).
 * gridDim.x = batch size
 * blockDim.x = 256 (tunable, must be a multiple of warpSize)
 *
 * For FP32 we load four floats at a time using float4.
 * For any other type we fall back to scalar loads.
 */
template <typename scalar_t>
__global__ void fused_reduce_kernel(
        const scalar_t* __restrict__ inp,   // (B, F)
        scalar_t* __restrict__ out,         // (B,)
        const int out_features) {

    const int batch_idx = blockIdx.x;
    const int tid       = threadIdx.x;
    const int lane      = tid & (warpSize - 1);
    const int warp_id   = tid / warpSize;
    const int warps     = blockDim.x / warpSize;

    const scalar_t* row_ptr = inp + static_cast<int64_t>(batch_idx) * out_features;

    // -------------------------------------------------
    // Vectorised reduction (float4) – only for FP32
    // -------------------------------------------------
    scalar_t thread_sum = static_cast<scalar_t>(0);

    // Size of a vector load (four scalars)
    constexpr int vec_width = 4;
    const int vec_len   = out_features / vec_width;                // whole vectors
    const int tail_start = vec_len * vec_width;                    // remaining scalars

    if constexpr (std::is_same<scalar_t, float>::value) {
        // reinterpret the row as a float4 array; guaranteed alignment by torch
        const float4* row4 = reinterpret_cast<const float4*>(row_ptr);
        // each thread processes a strided subset of the vectorised portion
        for (int i = tid; i < vec_len; i += blockDim.x) {
            float4 v = row4[i];
            thread_sum += v.x + v.y + v.z + v.w;
        }
    }

    // -----------------------------------------------------------------
    // Process any remaining columns that do not fit into a float4 pack
    // -----------------------------------------------------------------
    for (int col = tail_start + tid; col < out_features; col += blockDim.x) {
        thread_sum += row_ptr[col];
    }

    // ---------- warp‑level reduction ----------
    scalar_t warp_sum = warp_reduce_sum(thread_sum);

    // ---------- shared‑memory reduction ----------
    __shared__ scalar_t shmem[32];        // max 32 warps per block
    if (lane == 0) shmem[warp_id] = warp_sum;
    __syncthreads();

    // ---------- final reduction by the first warp ----------
    scalar_t block_sum = static_cast<scalar_t>(0);
    if (warp_id == 0) {
        scalar_t val = (tid < warps) ? shmem[tid] : static_cast<scalar_t>(0);
        block_sum = warp_reduce_sum(val);
    }

    // ---------- write the result ----------
    if (tid == 0) {
        // The original chain of reductions collapses to a plain sum.
        out[batch_idx] = block_sum;
    }
}

// ------------------------------------------------------------------
// C++ wrapper (PyBind11)
// ------------------------------------------------------------------
void fused_reduce(torch::Tensor input, torch::Tensor output) {
    const int batch   = input.size(0);
    const int out_feat = input.size(1);

    constexpr int threads = 256;          // keep the launch config used before
    const int blocks = batch;             // one block per batch element

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(),
                               "fused_reduce_kernel",
        ([&] {
            fused_reduce_kernel<scalar_t><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                out_feat);
        })
    );

    // Synchronisation only for debugging / correctness checks.
    // Can be removed for pure performance.
    cudaDeviceSynchronize();
}
"""

# ------------------------------------------------------------------
# C++ (PyBind11) binding source
# ------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

// Forward declaration of the CUDA implementation
void fused_reduce(torch::Tensor input, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_reduce", &fused_reduce,
          "Fused reduction kernel with 4‑wide vector loads (float4)");
}
"""

# ------------------------------------------------------------------
# Build the inline extension
# ------------------------------------------------------------------
fused_ext = load_inline(
    name="fused_reduce_ext_vec",
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cflags=['-O3'],
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True,
    verbose=False,
)

# ------------------------------------------------------------------
# Model definition – identical API to the reference implementation
# ------------------------------------------------------------------
class Model(nn.Module):
    """
    Model performing a linear layer followed by a fused post‑linear reduction.
    The reduction now uses a vectorised (float4) load for higher memory
    bandwidth utilisation.
    """
    def __init__(self, in_features: int, out_features: int):
        super(Model, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (B, in_features) on CUDA

        Returns
        -------
        torch.Tensor, shape (B, 1) on CUDA
        """
        # Linear transformation (cuBLAS)
        x = self.linear(x)                       # (B, out_features)

        # Allocate output tensor (B,1) – keep the extra dimension for API compatibility
        out = torch.empty(x.size(0), 1, device=x.device, dtype=x.dtype)

        # Call the vectorised fused reduction kernel.
        # The kernel writes a (B,) tensor; we squeeze the trailing dim only
        # to match the expected layout.
        fused_ext.fused_reduce(x, out.squeeze(-1))
        return out


# ------------------------------------------------------------------
# Helper functions expected by the test harness
# ------------------------------------------------------------------
def get_inputs(batch_size=1024, in_features=8192):
    """
    Returns a list with a single input tensor allocated on the GPU.
    """
    return [torch.rand(batch_size, in_features, device='cuda')]

def get_init_inputs():
    """
    Returns the arguments needed to instantiate ModelNew.
    The harness overwrites these globals (in_features/out_features) as needed.
    """
    # placeholders – the test harness will provide actual values.
    return [8192, 8192]   # [in_features, out_features]


# ------------------------------------------------------------------
# Simple sanity‑check (executed only when run directly)
# ------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    batch_size   = 1024
    in_features  = 8192
    out_features = 8192

    model = Model(in_features, out_features).cuda()
    model.eval()
    with torch.no_grad():
        inp = torch.rand(batch_size, in_features, device='cuda')
        out = model(inp)
        print(f"output shape: {out.shape}, mean value: {out.mean().item():.6f}")

