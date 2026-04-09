# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260405_222929/code_4.py
# fused_model_optimized.py
# -------------------------------------------------------------
# High‑performance fused reduction model for RTX 2080Ti (CUDA 12.5,
# PyTorch 2.10).  The kernel uses a grid‑stride loop to reduce each
# output row to a scalar while keeping memory accesses fully
# coalesced and the reduction fully inside CUDA (no extra PyTorch
# ops).  All logic is contained in a single file – the CUDA source is
# compiled inline with torch.utils.cpp_extension.load_inline().
# -------------------------------------------------------------
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ------------------------------------------------------------------
# CUDA kernel: one block may process multiple rows (grid‑stride over
# rows) and each thread iterates over columns with a stride equal to
# the block size.  The reduction employs warp shuffles and a small
# shared‑memory buffer for the final aggregation.
# ------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__device__ __forceinline__ scalar_t warp_reduce_sum(scalar_t val) {
    // A classic warp shuffle reduction (assumes warpSize = 32)
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// ---------------------------------------------------------------
// fused_reduce_kernel
//   inp : (B, F) matrix, row‑major
//   out : (B,) vector  – one scalar per input row
//   F   : number of columns (out_features)
// ---------------------------------------------------------------
template <typename scalar_t>
__global__ void fused_reduce_kernel(
    const scalar_t* __restrict__ inp,
    scalar_t* __restrict__ out,
    const int F) {

    // ----------- grid‑stride over rows (batch) ----------
    const int batch_idx = blockIdx.x;
    const int total_batches = gridDim.x;
    const int tid = threadIdx.x;
    const int lane = tid & (warpSize - 1);
    const int warp_id = tid >> 5;               // tid / warpSize
    const int warps_per_block = blockDim.x / warpSize;

    // shared memory for per‑warp partial sums (max 32 warps per block)
    __shared__ scalar_t warp_sum[32];

    // Process one or more rows depending on grid size
    for (int row = batch_idx; row < total_batches; row += total_batches) {
        const scalar_t* row_ptr = inp + (int64_t)row * F;

        // ----------- each thread reduces a strided slice of the row ----------
        scalar_t thread_sum = static_cast<scalar_t>(0);
        for (int col = tid; col < F; col += blockDim.x) {
            thread_sum += row_ptr[col];
        }

        // ----------- warp‑level reduction ----------
        scalar_t warp_res = warp_reduce_sum(thread_sum);

        // ----------- store per‑warp results ----------
        if (lane == 0) warp_sum[warp_id] = warp_res;
        __syncthreads();

        // ----------- final reduction by the first warp ----------
        scalar_t block_sum = static_cast<scalar_t>(0);
        if (warp_id == 0) {
            scalar_t val = (tid < warps_per_block) ? warp_sum[tid] : static_cast<scalar_t>(0);
            block_sum = warp_reduce_sum(val);
        }

        // ----------- write result ----------
        if (tid == 0) out[row] = block_sum;
        __syncthreads();        // ensure next row does not interfere
    }
}

// ------------------------------------------------------------------
// C++ wrapper called from Python
// ------------------------------------------------------------------
void fused_reduce(torch::Tensor inp, torch::Tensor out) {
    const int B = inp.size(0);
    const int F = inp.size(1);

    const int threads = 256;                 // 256 = 8 warps → good occupancy
    const int blocks = B;                    // one block per row (grid‑stride handles overflow)

    AT_DISPATCH_FLOATING_TYPES(inp.scalar_type(), "fused_reduce_kernel", [&] {
        fused_reduce_kernel<scalar_t><<<blocks, threads>>>(
            inp.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            F);
    });
    // Optional synchronization for debugging; can be removed for max perf.
    cudaDeviceSynchronize();
}
"""

# ------------------------------------------------------------------
# C++ / PyBind11 binding (required by load_inline)
# ------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

// Forward declaration of the CUDA implementation
void fused_reduce(torch::Tensor inp, torch::Tensor out);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_reduce", &fused_reduce,
          "Fused row‑wise sum reduction (CUDA)");
}
"""

# Build the extension in‑line
fused_ext = load_inline(
    name="fused_reduce_ext",
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cflags=['-O3'],
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True,
    verbose=False,
)

# ------------------------------------------------------------------
# Model definition – API identical to the original ModelNew
# ------------------------------------------------------------------
class Model(nn.Module):
    """
    Linear layer followed by a fused row‑wise sum reduction.
    The reduction chain that previously consisted of sum → max →
    mean → logsumexp → logsumexp collapses to a pure sum for a
    single scalar, hence this kernel returns exactly the same
    numerical result (within floating‑point tolerance).
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args
        ----
        x : torch.Tensor of shape (B, in_features) on CUDA

        Returns
        -------
        torch.Tensor of shape (B, 1) on CUDA
        """
        # Linear transformation (cuBLAS)
        x = self.linear(x)                     # (B, out_features)

        # Output placeholder – keep the extra dimension for API compatibility
        out = torch.empty(x.size(0), 1, device=x.device, dtype=x.dtype)

        # Invoke the fused CUDA reduction – note the squeeze adds no
        # extra copy (it just changes the view).
        fused_ext.fused_reduce(x, out.squeeze(-1))
        return out

# ------------------------------------------------------------------
# Helper utilities used by the external test harness
# ------------------------------------------------------------------
def get_inputs():
    """
    Returns a list containing a single input tensor of shape
    (batch_size, in_features) already on the GPU.
    The batch size and feature dimensions are expected to be set
    globally before calling this function.
    """
    return [torch.rand(batch_size, in_features, device='cuda')]

def get_init_inputs():
    """Return constructor arguments for ModelNew."""
    return [in_features, out_features]

# ------------------------------------------------------------------
# Simple sanity‑check (executed only when run directly)
# ------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    # Example dimensions – feel free to modify for testing
    batch_size = 1024
    in_features = 8192
    out_features = 8192

    model = Model(in_features, out_features).cuda()
    model.eval()
    with torch.no_grad():
        inp = torch.rand(batch_size, in_features, device='cuda')
        out = model(inp)
        print(f"output shape: {out.shape}, mean value: {out.mean().item():.6f}")

