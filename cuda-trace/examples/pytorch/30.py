# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260405_230857/code_1.py
# fused_model.py
# ------------------------------------------------------------
#   Fusion of ReLU + scalar division into a single CUDA kernel.
#   Optimization: Loop unrolling (#pragma unroll) – factor 4.
#   All operations run on the GPU (no CPU fallback).
# ------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# ------------------------------------------------------------------
# 1. CUDA kernel with loop unrolling
# ------------------------------------------------------------------
cuda_source = r'''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#ifndef UNROLL_FACTOR
#define UNROLL_FACTOR 4          // elements processed per thread per iteration
#endif

template <typename scalar_t>
__global__ void fused_relu_div_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const scalar_t divisor,
    const size_t N)               // total number of elements
{
    // Base thread index (covers the first UNROLL_FACTOR elements)
    const size_t base = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x * UNROLL_FACTOR;

    // ------------------------------------------------------------------
    // Unrolled body – each thread processes UNROLL_FACTOR consecutive items.
    // ------------------------------------------------------------------
    #pragma unroll
    for (int u = 0; u < UNROLL_FACTOR; ++u) {
        size_t idx = base + u * blockDim.x * gridDim.x;
        if (idx < N) {
            scalar_t val = input[idx];
            // ReLU
            val = val > static_cast<scalar_t>(0) ? val : static_cast<scalar_t>(0);
            // Division
            output[idx] = val / divisor;
        }
    }

    // ------------------------------------------------------------------
    // Handle any remaining tail elements when N is not a multiple of UNROLL_FACTOR.
    // ------------------------------------------------------------------
    for (size_t i = base + UNROLL_FACTOR * blockDim.x * gridDim.x;
         i < N; i += stride) {
        scalar_t val = input[i];
        val = val > static_cast<scalar_t>(0) ? val : static_cast<scalar_t>(0);
        output[i] = val / divisor;
    }
}

// ------------------------------------------------------------------
// C++ entry point called from Python
// ------------------------------------------------------------------
torch::Tensor fused_relu_div(torch::Tensor input, const double divisor) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");

    auto output = torch::empty_like(input);
    const size_t N = input.numel();

    // ------------------------------------------------------------------
    // Launch configuration – take unroll factor into account.
    // ------------------------------------------------------------------
    const int threads = 256;
    const int blocks  = (N + threads * UNROLL_FACTOR - 1) / (threads * UNROLL_FACTOR);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(),
        "fused_relu_div_kernel",
        [&] {
            fused_relu_div_kernel<scalar_t><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                static_cast<scalar_t>(divisor),
                N);
        });

    // Propagate any launch errors.
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA kernel error: ") +
                                 cudaGetErrorString(err));
    }
    return output;
}
'''

cpp_source = r'''
#include <torch/extension.h>

// Forward declaration of the kernel wrapper.
torch::Tensor fused_relu_div(torch::Tensor input, const double divisor);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_relu_div", &fused_relu_div,
          "Fused ReLU + division kernel with loop unrolling");
}
'''

# Build the extension once on import.
_fused_ext = load_inline(
    name="fused_relu_div_ext",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cflags=['-O3'],
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True,
    verbose=False,
)

# ------------------------------------------------------------------
# 2. Model that uses the fused kernel
# ------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Linear layer followed by a fused ReLU+division kernel (loop‑unrolled).
    All heavy work stays on the GPU.
    """
    def __init__(self, in_features: int, out_features: int, divisor: float):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)
        self.divisor = divisor

        # Move parameters to GPU once during construction.
        self.linear.weight.data = self.linear.weight.data.cuda()
        if self.linear.bias is not None:
            self.linear.bias.data = self.linear.bias.data.cuda()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input lives on GPU.
        x = x.cuda()

        # GEMM via cuBLAS (torch.nn.functional.linear).
        out = F.linear(x, self.linear.weight, self.linear.bias)

        # Fused ReLU + division with loop‑unrolled kernel.
        out = _fused_ext.fused_relu_div(out, self.divisor)

        return out


# ------------------------------------------------------------------
# 3. Helper functions – keep identical signatures to the original script.
# ------------------------------------------------------------------
batch_size = 1024
in_features = 8192
out_features = 8192
divisor = 2.0

def get_inputs():
    """Return a list with a single random tensor already on CUDA."""
    return [torch.rand(batch_size, in_features, device='cuda')]

def get_init_inputs():
    """Return constructor arguments for ModelNew."""
    return [in_features, out_features, divisor]


# ------------------------------------------------------------------
# 4. Simple sanity test / timing when run directly.
# ------------------------------------------------------------------
if __name__ == "__main__":
    torch.cuda.empty_cache()
    model = ModelNew(*get_init_inputs())
    inp = get_inputs()[0]

    # Warm‑up passes.
    with torch.no_grad():
        for _ in range(5):
            _ = model(inp)

    torch.cuda.synchronize()
    import time
    t0 = time.time()
    with torch.no_grad():
        out = model(inp)
    torch.cuda.synchronize()
    print(f"Elapsed time (ms): {(time.time() - t0) * 1000:.3f}")
    print("Output shape:", out.shape)
