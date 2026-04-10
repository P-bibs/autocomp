# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_143434/code_10.py
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




# -*- coding: utf-8 -*-
# ------------------------------------------------------------
#  High‑performance reduction of a (B, D1, D2) tensor along dim‑1
#  using a grid‑stride kernel and float4 vector loads.
# ------------------------------------------------------------
import torch
from torch.utils.cpp_extension import load_inline

# -------------------------------------------------------------------------
#  CUDA source – grid‑stride kernel with float4 loads
# -------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// ------------------------------------------------------------
//  Kernel: each thread processes many output elements via a
//  grid‑stride loop.  For the aligned case we load a float4,
//  otherwise we fall back to scalar loads.
// ------------------------------------------------------------
__global__ void sum_dim1_gridstride_kernel(const float* __restrict__ input,
                                           float* __restrict__ output,
                                           const int B,
                                           const int D1,
                                           const int D2)
{
    // ----- 1. linear index of the (b, j) output element -----
    // total number of output elements = B * D2
    const int total = B * D2;
    const int tid   = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    for (int linear = tid; linear < total; linear += stride) {
        const int b = linear / D2;          // batch index
        const int j = linear % D2;          // column inside the batch

        // --------------------------------------------------------
        // 2. accumulate over the D1 dimension
        // --------------------------------------------------------
        float sum = 0.0f;

        // Fast path – the column is 4‑aligned and there are at least
        // 4 elements left in D2.  We can safely reinterpret the address
        // as a float4* and load four floats in one transaction.
        if ( (j % 4 == 0) && (j + 3 < D2) ) {
            // we will load D1 elements, each of them a float4.
            // The pointer arithmetic is the same as in the original code.
            const float4* src = reinterpret_cast<const float4*>(
                input + b * D1 * D2 + j);
            float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);

            #pragma unroll 4
            for (int i = 0; i < D1; ++i) {
                // __ldg is a read‑only cache‑load (L1) – it is cheap.
                const float4 v = __ldg(src + i * (D2 / 4));
                acc.x += v.x;
                acc.y += v.y;
                acc.z += v.z;
                acc.w += v.w;
            }
            // Fold the four components back into a single scalar sum.
            sum = acc.x + acc.y + acc.z + acc.w;
        }
        else {
            // General case – scalar loads.  This path is only taken for
            // the few columns at the end of each row that are not 4‑aligned.
            const float* src = input + b * D1 * D2 + j;
            #pragma unroll 4
            for (int i = 0; i < D1; ++i) {
                sum += __ldg(src + i * D2);
            }
        }

        // ----- 3. write the result -----
        output[linear] = sum;
    }
}

// ------------------------------------------------------------
//  Host wrapper
// ------------------------------------------------------------
void sum_dim1_gridstride(torch::Tensor input, torch::Tensor output)
{
    const int B  = input.size(0);
    const int D1 = input.size(1);
    const int D2 = input.size(2);

    // Choose a modest block size – 256 works well on RTX 2080 Ti.
    // The grid is sized so that we have at least one block per 1024
    // output elements, but the kernel can also handle any oversubscription
    // thanks to the grid‑stride loop.
    const int threads = 256;
    const int blocks  = (B * D2 + threads - 1) / threads;

    sum_dim1_gridstride_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        B, D1, D2
    );
}
"""

# -------------------------------------------------------------------------
#  C++ binding (PYBIND11)
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

// Declaration of the host‑side launcher defined in the .cu file.
void sum_dim1_gridstride(torch::Tensor input, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sum_dim1_gridstride",
          &sum_dim1_gridstride,
          "Sum along dimension 1 (grid‑stride version, float4 loads)");
}
"""

# Compile the extension (the flags already contain -O3 and --use_fast_math)
_sum_ext = load_inline(
    name='sum_dim1_gridstride_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True,
    verbose=False
)

# -------------------------------------------------------------------------
#  Functional model – pure‑Python API
# -------------------------------------------------------------------------
def functional_model(x: torch.Tensor, *, dim: int) -> torch.Tensor:
    """
    Reduce a 3‑D tensor ``x`` of shape (B, D1, D2) along dimension 1
    using the custom CUDA kernel defined above.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor (must be on CUDA, dtype float32).
    dim : int
        Only ``dim == 1`` is supported – this mirrors the original code.

    Returns
    -------
    torch.Tensor
        Tensor of shape (B, 1, D2) containing the sums.
    """
    assert dim == 1, "Only reduction along dimension 1 is supported."
    assert x.is_cuda, "Input tensor must be on CUDA."
    assert x.dtype == torch.float32, "Only float32 is supported."

    B, D1, D2 = x.shape

    # Allocate the output (B, D2) – the kernel writes a flat array.
    out = torch.empty((B, D2), device=x.device, dtype=x.dtype)

    # Launch the optimized kernel.
    _sum_ext.sum_dim1_gridstride(x, out)

    # The public API expects a kept‑dim (B, 1, D2) tensor.
    return out.unsqueeze(1)


# -------------------------------------------------------------------------
#  Quick sanity‑check (executed only when the file is run directly)
# -------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    batch_size = 128
    dim1 = 4096
    dim2 = 4095
    reduce_dim = 1

    x = torch.rand(batch_size, dim1, dim2,
                   device='cuda',
                   dtype=torch.float32,
                   pin_memory=False)

    # Run the model
    out = functional_model(x, dim=reduce_dim)

    # Verify shape
    assert out.shape == (batch_size, 1, dim2), f"wrong shape: {out.shape}"

    # Verify numerical correctness (tolerance 1e‑4 matches the original test)
    ref = x.sum(dim=1, keepdim=True)
    if not torch.allclose(out, ref, atol=1e-4):
        max_err = (out - ref).abs().max()
        raise RuntimeError(f"Result mismatch! max error = {max_err}")
    print("✓ functional_model with grid‑stride kernel works correctly.")
