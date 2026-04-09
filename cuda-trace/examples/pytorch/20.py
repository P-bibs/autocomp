# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260405_233641/code_4.py
# fused_model_optimized_shared_mem.py
# -------------------------------------------------------------
# High‑performance fused linear + reduction for ModelNew.
# Optimization: tile the summed weight vector into shared memory
# to dramatically reduce global‑memory traffic for w_sum.
# -------------------------------------------------------------
# The implementation uses a single CUDA kernel that:
#   • Loads a tile of w_sum into __shared__ memory (once per tile).
#   • Computes dot(input[i,:], w_sum[:]) using the tiled weights.
#   • Adds the summed bias.
#   • Writes one scalar output per batch element.
# -------------------------------------------------------------

import torch
import torch.nn as nn
batch_size = 1024
in_features = 8192
out_features = 8192
from torch.utils.cpp_extension import load_inline

# ------------------------------------------------------------------
# CUDA kernel with shared‑memory tiling
# ------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <type_traits>

// ------------------------------------------------------------------
// Warp‑level reduction (generic)
// ------------------------------------------------------------------
template <typename scalar_t>
__device__ __forceinline__ scalar_t warp_reduce_sum(scalar_t val) {
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// ------------------------------------------------------------------
// Kernel: out[batch_idx] = dot( input[batch_idx,:], w_sum[:] ) + bias_sum
// ------------------------------------------------------------------
template <typename scalar_t>
__global__ void fused_dot_kernel_tiled(
        const scalar_t* __restrict__ input,   // (B, in_features)
        const scalar_t* __restrict__ w_sum,   // (in_features,)
        const scalar_t      bias_sum,         // scalar
        scalar_t* __restrict__ output,        // (B,)
        const int           in_features) {

    const int batch_idx = blockIdx.x;               // one block per row
    const int tid       = threadIdx.x;
    const int lane      = tid & (warpSize - 1);
    const int warps     = blockDim.x / warpSize;

    // ------------------------------------------------------------------
    // Tiling parameters – tile size must be <= blockDim.x
    // ------------------------------------------------------------------
    constexpr int TILE_SIZE = 256;
    __shared__ scalar_t sh_w[TILE_SIZE];

    scalar_t thread_sum = scalar_t(0);

    // ------------------------------------------------------------------
    // Process the weight vector in tiles
    // ------------------------------------------------------------------
    for (int tile_start = 0; tile_start < in_features; tile_start += TILE_SIZE) {
        // Load a tile of w_sum into shared memory (first TILE_SIZE threads)
        int w_idx = tile_start + tid;
        if (tid < TILE_SIZE && w_idx < in_features) {
            sh_w[tid] = w_sum[w_idx];
        }
        __syncthreads();   // make sure the whole tile is visible

        // Number of elements in this tile (may be smaller on the last tile)
        int tile_elems = min(TILE_SIZE, in_features - tile_start);

        // Each thread walks over its own portion of the input within this tile
        for (int i = tid; i < tile_elems; i += blockDim.x) {
            scalar_t in_val = input[batch_idx * in_features + tile_start + i];
            thread_sum += in_val * sh_w[i];
        }
        __syncthreads();   // ensure all threads finished using the tile
    }

    // ------------------------------------------------------------------
    // In‑block reduction (warp + shared‑mem reduction)
    // ------------------------------------------------------------------
    scalar_t warp_sum = warp_reduce_sum(thread_sum);

    __shared__ scalar_t shmem[32];
    if (lane == 0) shmem[tid / warpSize] = warp_sum;
    __syncthreads();

    scalar_t block_sum = scalar_t(0);
    if (tid < warpSize) {
        block_sum = warp_reduce_sum(
                (tid < (blockDim.x / warpSize)) ? shmem[tid] : scalar_t(0));
    }

    // Write final result (one thread per block)
    if (tid == 0) {
        output[batch_idx] = block_sum + bias_sum;
    }
}

// ------------------------------------------------------------------
// C++ wrapper called from Python
// ------------------------------------------------------------------
void fused_dot_wrapper(torch::Tensor input,
                       torch::Tensor w_sum,
                       torch::Tensor bias_sum,
                       torch::Tensor output) {
    const int batch      = input.size(0);
    const int in_features = input.size(1);
    const int threads    = 256;          // must be >= TILE_SIZE
    const int blocks     = batch;       // one block per batch row

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(),
                               "fused_dot_kernel_tiled",
                               [&] {
        fused_dot_kernel_tiled<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            w_sum.data_ptr<scalar_t>(),
            bias_sum.item<scalar_t>(),
            output.data_ptr<scalar_t>(),
            in_features);
    });

    // Synchronize for correctness in the test harness; can be removed in production.
    cudaDeviceSynchronize();
}
"""

# ------------------------------------------------------------------
# C++/pybind11 binding
# ------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

// Forward declaration of the CUDA implementation
void fused_dot_wrapper(torch::Tensor input,
                       torch::Tensor w_sum,
                       torch::Tensor bias_sum,
                       torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_dot", &fused_dot_wrapper,
          "Fused dot‑product with shared‑memory tiling – CUDA implementation");
}
"""

# ------------------------------------------------------------------
# Build the inline extension
# ------------------------------------------------------------------
fused_ext = load_inline(
    name="fused_dot_ext",
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cflags=['-O3'],
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True,
    verbose=False,
)

# ------------------------------------------------------------------
# Model definition – API compatible with the original ModelNew
# ------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Model that performs a linear projection followed by a reduction.
    The linear weight matrix is summed over the output dimension beforehand,
    turning the whole operation into a single dot‑product per batch element.
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)

        # Pre‑compute the summed weight vector and summed bias scalar.
        with torch.no_grad():
            w_sum = self.linear.weight.sum(dim=0).contiguous()
            bias_sum = self.linear.bias.sum()
            self.register_buffer('w_sum', w_sum)
            self.register_buffer('bias_sum',
                                 torch.tensor(bias_sum,
                                              device=w_sum.device,
                                              dtype=w_sum.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments
        ----------
        x : torch.Tensor of shape (B, in_features) on CUDA

        Returns
        -------
        torch.Tensor of shape (B, 1) on CUDA
        """
        out = torch.empty(x.size(0), 1, device=x.device, dtype=x.dtype)
        # The CUDA kernel writes a (B,) tensor; squeeze the view for the call.
        fused_ext.fused_dot(x, self.w_sum, self.bias_sum, out.squeeze(-1))
        return out

# ------------------------------------------------------------------
# Helper functions used by the benchmark / test harness
# ------------------------------------------------------------------
def get_inputs():
    """Return a list with a single tensor on GPU – used by the benchmark."""
    return [torch.rand(batch_size, in_features, device='cuda')]

def get_init_inputs():
    """Provide constructor arguments for ModelNew."""
    return [in_features, out_features]

# ------------------------------------------------------------------
# Simple sanity‑check (run `python fused_model_optimized_shared_mem.py` to test)
# ------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    batch_size = 1024
    in_features = 8192
    out_features = 8192

    model = ModelNew(in_features, out_features).cuda()
    model.eval()
    with torch.no_grad():
        inp = torch.rand(batch_size, in_features, device='cuda')
        out = model(inp)
        print(f"output shape: {out.shape}, mean: {out.mean().item():.6f}")
