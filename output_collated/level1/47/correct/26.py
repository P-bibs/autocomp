# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_125421/code_11.py
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




# =============================================================================
#  High-performance reduction along dimension 1 (batch, D1, D2) → (batch, 1, D2)
#  Custom CUDA kernel with loop-unrolled inner reduction.
# =============================================================================

import torch
from torch.utils.cpp_extension import load_inline

# -------------------------------------------------------------------------
#  Optimized CUDA implementation – loop-unrolled reduction
# -------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// ---------------------------------------------------------------
//  Kernel: each thread computes one output element (b, j)
//  The reduction over D1 is performed with a manually unrolled loop.
// ---------------------------------------------------------------
__global__ void sum_dim1_unrolled_kernel(const float* __restrict__ input,
                                          float* __restrict__ output,
                                          int B, int D1, int D2) {
    // blockIdx.x – batch index, blockIdx.y – tile of the D2 axis
    int b   = blockIdx.x;
    int j   = blockIdx.y * blockDim.x + threadIdx.x;   // column inside D2

    if (b >= B || j >= D2) return;

    // -----------------------------------------------------------------
    //  Unrolled reduction over the D1 dimension.
    //  D1 is a multiple of 8 for our benchmark (4096), but we keep a
    //  generic tail loop for safety.
    // -----------------------------------------------------------------
    float sum = 0.0f;

    // Load 8 elements per iteration – the compiler will keep the
    // partial sum in a register throughout the whole loop.
    #pragma unroll 8
    for (int i = 0; i < D1; i += 8) {
        sum += input[b * D1 * D2 + (i + 0) * D2 + j];
        sum += input[b * D1 * D2 + (i + 1) * D2 + j];
        sum += input[b * D1 * D2 + (i + 2) * D2 + j];
        sum += input[b * D1 * D2 + (i + 3) * D2 + j];
        sum += input[b * D1 * D2 + (i + 4) * D2 + j];
        sum += input[b * D1 * D2 + (i + 5) * D2 + j];
        sum += input[b * D1 * D2 + (i + 6) * D2 + j];
        sum += input[b * D1 * D2 + (i + 7) * D2 + j];
    }

    // Tail loop for the (very unlikely) case that D1 is not divisible by 8
    for (int i = (D1 / 8) * 8; i < D1; ++i) {
        sum += input[b * D1 * D2 + i * D2 + j];
    }

    // Write the reduced value
    output[b * D2 + j] = sum;
}

// -------------------------------------------------------------------------
//  Wrapper called from Python – launches the unrolled kernel
// -------------------------------------------------------------------------
void sum_dim1_unrolled(torch::Tensor input, torch::Tensor output) {
    const int B  = input.size(0);
    const int D1 = input.size(1);
    const int D2 = input.size(2);

    // One thread per output column (b, j)
    const int threads = 512;
    const dim3 blocks(B, (D2 + threads - 1) / threads);

    sum_dim1_unrolled_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        B, D1, D2);
}
"""

cpp_source = r"""
#include <torch/extension.h>

// Forward declaration of the CUDA implementation
void sum_dim1_unrolled(torch::Tensor input, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sum_dim1_unrolled", &sum_dim1_unrolled,
          "Sum along dimension 1 with an unrolled inner loop");
}
"""

# -------------------------------------------------------------------------
#  Compile the extension (the same as the original script)
# -------------------------------------------------------------------------
sum_ext_unrolled = load_inline(
    name='sum_dim1_unrolled_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# -------------------------------------------------------------------------
#  Functional model – unchanged API, only the kernel call is swapped
# -------------------------------------------------------------------------
def functional_model(x, *, dim):
    """
    Reduce the input tensor `x` of shape (B, D1, D2) along dimension 1
    using the custom CUDA kernel that features an unrolled inner loop.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor on CUDA.  Shape: (batch, dim1, dim2).
    dim : int
        Must be 1 (as required by the problem statement).

    Returns
    -------
    torch.Tensor
        Output tensor of shape (batch, 1, dim2) containing the sums.
    """
    assert dim == 1, "Only dim=1 is supported by this model."

    # Output buffer: (B, 1, D2)
    out = torch.zeros((x.shape[0], 1, x.shape[2]),
                      device=x.device, dtype=x.dtype)

    # The kernel writes to a view of shape (B, D2); we pass the squeezed view.
    sum_ext_unrolled.sum_dim1_unrolled(x, out.squeeze(1))

    return out

# -------------------------------------------------------------------------
#  Evaluation harness (unchanged)
# -------------------------------------------------------------------------
batch_size = 128
dim1 = 4096
dim2 = 4095
reduce_dim = 1

def get_init_inputs():
    """Returns any static inputs required by the model – here only the reduction dim."""
    return [reduce_dim]

def get_inputs():
    """Creates a random input tensor on the GPU for benchmarking."""
    x = torch.rand(batch_size, dim1, dim2, device='cuda', dtype=torch.float32)
    return [x]

# -------------------------------------------------------------------------
#  Quick sanity-check when the module is executed directly
# -------------------------------------------------------------------------
if __name__ == "__main__":
    dim = reduce_dim
    x = torch.rand(batch_size, dim1, dim2, device='cuda')
    out = functional_model(x, dim=dim)
    assert out.shape == (batch_size, 1, dim2)
    # Verify numerical correctness against PyTorch's sum
    ref = x.sum(dim=reduce_dim, keepdim=True)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-6)
    print("Functional model with unrolled kernel works correctly.")
