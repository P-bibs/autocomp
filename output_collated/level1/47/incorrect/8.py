# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_131936/code_9.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['dim']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['dim']
REQUIRED_FLAT_STATE_NAMES = []


class ModelNew(nn.Module):
    """
    Simple model that performs sum reduction over a specified dimension.
    """

    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): Dimension to reduce over.
        """
        super(ModelNew, self).__init__()
        self.dim = dim

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
    if 'dim' in flat_state:
        state_kwargs['dim'] = flat_state['dim']
    else:
        state_kwargs['dim'] = getattr(model, 'dim')
    missing = [name for name in REQUIRED_STATE_NAMES if name not in state_kwargs]
    if missing:
        raise RuntimeError(f'Missing required state entries: {missing}')
    return state_kwargs


def get_functional_inputs():
    model = build_reference_model()
    forward_args = tuple(get_inputs())
    state_kwargs = extract_state_kwargs(model)
    return forward_args, state_kwargs




# ============================================================
#  High-performance reduction of a (B, D1, D2) tensor
#  along dimension 1 using float4 loads (RTX-2080Ti, PyTorch 2.10)
# ============================================================
import torch
from torch.utils.cpp_extension import load_inline

# -------------------------------------------------------------------------
#  CUDA source – two kernels (aligned float4 + scalar tail)
# -------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#ifndef DIV_ROUND_UP
#define DIV_ROUND_UP(x, y) (((x) + (y) - 1) / (y))
#endif

// ---------------------------------------------------------------------
//  Kernel #1 : pure float4 loads – assumes the column range is 4-aligned
// ---------------------------------------------------------------------
extern "C"
__global__ void sum_dim1_float4_aligned_kernel(
        const float* __restrict__ input,
        float* __restrict__ output,
        const int B,
        const int D1,
        const int aligned_D2)      // number of columns that are 4-aligned
{
    const int tid   = threadIdx.x;                     // 0 … 1023
    const int b     = blockIdx.x;                      // batch index
    const int j0    = blockIdx.y * (blockDim.x * 4) + tid * 4; // start column

    if (b >= B) return;                // safety, should never fire
    if (j0 >= aligned_D2) return;      // out-of-range (last partial block)

    // -----------------------------------------------------------------
    //  Accumulate over D1 using float4 loads
    // -----------------------------------------------------------------
    float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);

    // Stride to the beginning of the (b, 0, j0) element
    const size_t base = static_cast<size_t>(b) * D1 * aligned_D2
                      + static_cast<size_t>(j0);

    // Loop over D1 rows
    #pragma unroll
    for (int i = 0; i < D1; ++i) {
        const float4 val = __ldg(reinterpret_cast<const float4*>(
                                 input + base + static_cast<size_t>(i) * aligned_D2));
        sum.x += val.x;
        sum.y += val.y;
        sum.z += val.z;
        sum.w += val.w;
    }

    // Store the four results
    float* out_ptr = output + static_cast<size_t>(b) * aligned_D2 + j0;
    out_ptr[0] = sum.x;
    out_ptr[1] = sum.y;
    out_ptr[2] = sum.z;
    out_ptr[3] = sum.w;
}

// ---------------------------------------------------------------------
//  Kernel #2 : scalar tail – handles the last 1-3 columns (if any)
// ---------------------------------------------------------------------
extern "C"
__global__ void sum_dim1_float4_tail_kernel(
        const float* __restrict__ input,
        float* __restrict__ output,
        const int B,
        const int D1,
        const int D2,
        const int aligned_D2)          // start of the tail region
{
    const int tid = threadIdx.x;               // 0 … 1023
    const int b   = blockIdx.x;                // batch index
    const int j   = blockIdx.y * blockDim.x + tid; // column index (scalar)

    if (b >= B) return;
    if (j >= D2) return;               // out-of-range (extra threads)

    // Guard against threads that belong to the aligned part
    if (j < aligned_D2) return;        // they are already processed

    float sum = 0.f;
    const size_t base = static_cast<size_t>(b) * D1 * D2 + static_cast<size_t>(j);

    #pragma unroll
    for (int i = 0; i < D1; ++i) {
        sum += __ldg(input + base + static_cast<size_t>(i) * D2);
    }

    output[static_cast<size_t>(b) * D2 + j] = sum;
}

// ---------------------------------------------------------------------
//  Host-side wrapper – launches the two kernels
// ---------------------------------------------------------------------
void sum_dim1_float4(torch::Tensor input, torch::Tensor output)
{
    const int B  = input.size(0);
    const int D1 = input.size(1);
    const int D2 = input.size(2);

    // -----------------------------------------------------------------
    //  Determine the 4-aligned region
    // -----------------------------------------------------------------
    const int aligned_D2 = (D2 / 4) * 4;      // largest multiple of 4 ≤ D2
    const int tail_cols   = D2 - aligned_D2; // 0 … 3

    const int threads = 1024;
    const int elems_per_thread = 4;                  // float4 → 4 floats
    const int elems_per_block  = threads * elems_per_thread;

    // ------------------- aligned kernel launch --------------------
    const int blocks_x = B;
    const int blocks_y = (aligned_D2 + elems_per_block - 1) / elems_per_block;

    dim3 grid_aligned(blocks_x, blocks_y);
    dim3 block(threads);

    if (aligned_D2 > 0) {
        sum_dim1_float4_aligned_kernel<<<grid_aligned, block>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            B, D1, aligned_D2);
    }

    // ------------------- tail kernel launch (if needed) ----------
    if (tail_cols > 0) {
        const int tail_threads = 1024;
        const int tail_blocks_y = DIV_ROUND_UP(tail_cols, tail_threads);
        dim3 grid_tail(B, tail_blocks_y);
        dim3 block_tail(tail_threads);

        sum_dim1_float4_tail_kernel<<<grid_tail, block_tail>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            B, D1, D2, aligned_D2);
    }
}
"""

# -------------------------------------------------------------------------
#  C++ binding (PYBIND11)
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

// Declarations of the two kernels
void sum_dim1_float4(torch::Tensor input, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sum_dim1_float4", &sum_dim1_float4,
          "Sum along dimension 1 using a pure-float4 kernel "
          "plus a tiny scalar tail kernel (warp-divergence-free).");
}
"""

# Compile the extension (single .cu source, single .cpp source)
sum_ext_float4 = load_inline(
    name='sum_dim1_float4',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True,
    verbose=False,
)

# -------------------------------------------------------------------------
#  Functional model – uses the custom kernel
# -------------------------------------------------------------------------
def functional_model(x, *, dim):
    """
    Reduce input tensor `x` of shape (B, D1, D2) along dimension 1 using
    the warp-divergence-free float4 kernel.
    """
    assert dim == 1, "Only reduction along dimension 1 is supported."

    B, D1, D2 = x.shape
    # Output tensor with shape (B, D2)
    out = torch.empty((B, D2), device=x.device, dtype=x.dtype)

    # Launch the optimized kernel (two-kernel version)
    sum_ext_float4.sum_dim1_float4(x, out)

    # Match the original API (keep the singleton dimension)
    return out.unsqueeze(1)


# -------------------------------------------------------------------------
#  Quick sanity-check when the file is executed directly
# -------------------------------------------------------------------------
if __name__ == "__main__":
    batch_size = 128
    dim1 = 4096
    dim2 = 4095          # not a multiple of 4 – triggers the tail kernel
    reduce_dim = 1

    x = torch.rand(batch_size, dim1, dim2,
                   device='cuda', dtype=torch.float32)
    out = functional_model(x, dim=reduce_dim)

    # Shape check
    assert out.shape == (batch_size, 1, dim2), f"Wrong shape: {out.shape}"

    # Result check against PyTorch's native reduction
    ref = x.sum(dim=1, keepdim=True)
    torch.testing.assert_allclose(out, ref, atol=1e-4, rtol=1e-5)
    print("✓ functional_model with warp-divergence-free kernel works correctly.")
