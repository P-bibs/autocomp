# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260405_230857/code_2.py
# fused_model.py
# --------------------------------------------------------------
# Optimised model – in‑place fused ReLU + division (RTX 2080 Ti)
# --------------------------------------------------------------
# This file is a drop‑in replacement for the original script.
# Only the class ``ModelNew`` is expected to be imported during
# evaluation. All tensor work happens on CUDA and the post‑linear
# ReLU/division is performed entirely in‑place, cutting global‑memory
# traffic by ~50 %.
# --------------------------------------------------------------

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ------------------------------------------------------------------
# 1) Inline CUDA extension – in‑place fused ReLU + division kernel
# ------------------------------------------------------------------
# The kernel reads each element, applies ReLU, divides by ``divisor``,
# and writes back to the same address (read‑modify‑write).
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// ------------------------------------------------------------------
// In‑place fused kernel: ReLU then divide by a constant.
// Works with float, double and half (via AT_DISPATCH).
// ------------------------------------------------------------------
template <typename scalar_t>
__global__ void fused_relu_div_inplace_kernel(
    scalar_t* __restrict__ data,       // single pointer – operation is in‑place
    const scalar_t divisor,
    const size_t N)                    // total number of elements
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        scalar_t v = data[idx];
        // ReLU
        v = (v > static_cast<scalar_t>(0)) ? v : static_cast<scalar_t>(0);
        // Division (fast‑math enabled by compile flag)
        data[idx] = v / divisor;
    }
}

// ------------------------------------------------------------------
// C++ wrapper that launches the kernel.
// ------------------------------------------------------------------
void fused_relu_div_inplace(torch::Tensor data, double divisor) {
    const size_t N = data.numel();
    const int threads = 256;
    const int blocks = static_cast<int>((N + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        data.scalar_type(),
        "fused_relu_div_inplace",
        [&] {
            fused_relu_div_inplace_kernel<scalar_t><<<blocks, threads>>>(
                data.data_ptr<scalar_t>(),
                static_cast<scalar_t>(divisor),
                N);
        });

    // Propagate any launch errors to Python.
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("CUDA kernel failed: ", cudaGetErrorString(err));
    }
}
"""

cpp_source = r"""
#include <torch/extension.h>

// Forward declaration of the function defined in the .cu file.
void fused_relu_div_inplace(torch::Tensor data, double divisor);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_relu_div_inplace", &fused_relu_div_inplace,
          "Fused ReLU + division in‑place (CUDA)");
}
"""

# Compile the extension at import time.  ``verbose=False`` keeps the import clean.
_fused_ext = load_inline(
    name="fused_relu_div_ext",
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    with_cuda=True,
    verbose=False,
)

# ------------------------------------------------------------------
# 2) Model definition – uses the in‑place fused kernel.
# ------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Functionally identical to the original model:
      * Linear layer (cuBLAS GEMM)
      * ReLU + division, performed in‐place via a custom CUDA kernel.
    All tensors live on the GPU, eliminating extra output buffers.
    """
    def __init__(self, in_features: int, out_features: int, divisor: float):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)
        self.divisor = divisor
        # Force parameters onto the GPU once at construction.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Put input on the correct device (non‑blocking copy if needed).
        x = x.to(self.device, non_blocking=True)

        # Linear layer – uses cuBLAS.
        x = self.linear(x)

        # In‑place fused ReLU + division.
        _fused_ext.fused_relu_div_inplace(x, self.divisor)

        return x

# ------------------------------------------------------------------
# 3) Helper functions – unchanged public API.
# ------------------------------------------------------------------
batch_size = 1024
in_features = 8192
out_features = 8192
divisor = 2.0

def get_inputs():
    """Return a list containing a single input tensor placed on the GPU."""
    return [torch.rand(batch_size, in_features, device="cuda")]

def get_init_inputs():
    """Return the arguments required to instantiate ModelNew."""
    return [in_features, out_features, divisor]

# ------------------------------------------------------------------
# 4) Simple sanity‑check (executed only when run directly).
# ------------------------------------------------------------------
if __name__ == "__main__":
    model = ModelNew(*get_init_inputs())
    model.eval()
    with torch.no_grad():
        X = get_inputs()[0]
        Y = model(X)
        print("Output shape:", Y.shape, "Device:", Y.device)

        # Reference calculation using ordinary PyTorch ops.
        torch_ref = torch.relu(model.linear(X)) / divisor
        max_diff = (Y - torch_ref).abs().max().item()
        print("Maximum absolute difference vs reference:", max_diff)
