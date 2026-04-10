# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_122232/code_7.py
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




import torch
from torch.utils.cpp_extension import load_inline

# -------------------------------------------------------------------------
# Inline CUDA source – the only difference is the #pragma unroll 16
# and the hoisted base pointer (input + b*D1*D2).
# -------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void sum_dim1_kernel(const float* __restrict__ input,
                                float* __restrict__ output,
                                const int B,
                                const int D1,
                                const int D2) {
    // Block / thread indices
    const int b = blockIdx.x;
    const int j = blockIdx.y * blockDim.x + threadIdx.x;

    // Bounds check – early exit for the last partially‑filled block
    if (b < B && j < D2) {
        // Hoist the base pointer for this batch element.
        const float* __restrict__ in = input + b * (D1 * D2);

        float sum = 0.0f;

        // ---- Optimization: unroll the reduction loop -----------------
        #pragma unroll 16
        // ----------------------------------------------------------------
        for (int i = 0; i < D1; ++i) {
            // Contiguous memory access for a fixed i, varying j (coalesced)
            sum += in[i * D2 + j];
        }

        // Write the result – output shape is (B, 1, D2) but stored as
        // a flattened (B*D2) tensor, which matches index b*D2 + j.
        output[b * D2 + j] = sum;
    }
}

void sum_dim1(torch::Tensor input, torch::Tensor output) {
    const int B  = input.size(0);
    const int D1 = input.size(1);
    const int D2 = input.size(2);

    const dim3 threads(256);
    const dim3 blocks(B, (D2 + threads.x - 1) / threads.x);

    sum_dim1_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        B, D1, D2);
}
"""

# -------------------------------------------------------------------------
# C++ binding (PYBIND11)
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void sum_dim1(torch::Tensor input, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sum_dim1", &sum_dim1, "Sum along dimension 1");
}
"""

# -------------------------------------------------------------------------
# Compile the extension with the same compiler flags as before
# -------------------------------------------------------------------------
sum_ext = load_inline(
    name='sum_dim1',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True,
)

# -------------------------------------------------------------------------
# Functional wrapper – the only entry point that will be imported.
# -------------------------------------------------------------------------
def functional_model(x, *, dim):
    """
    Sums tensor `x` over the given dimension `dim`.
    Only dim==1 is supported (reduction of the middle dimension).
    """
    if dim != 1:
        raise ValueError("Only reduction over dimension 1 is implemented")
    # Output shape: (batch_size, 1, dim2)
    output = torch.zeros((x.shape[0], 1, x.shape[2]),
                         device=x.device, dtype=x.dtype)
    sum_ext.sum_dim1(x, output)
    return output


# -------------------------------------------------------------------------
# Evaluation helpers (not part of the functional model, but required for
# the benchmark harness).
# -------------------------------------------------------------------------
batch_size = 128
dim1 = 4096
dim2 = 4095

def get_init_inputs():
    return [1]  # reduce over dim 1

def get_inputs():
    # Random input on the GPU
    return [torch.rand(batch_size, dim1, dim2, device='cuda')]

# -------------------------------------------------------------------------
# The file may be imported; only `functional_model` will be called.
# -------------------------------------------------------------------------
if __name__ == "__main__":
    # Quick sanity‑check
    x = torch.rand(2, 4, 5, device='cuda')
    y = functional_model(x, dim=1)
    print("Output shape:", y.shape)
    # Expected: (2,1,5)
    # Verify manually
    expected = x.sum(dim=1, keepdim=True)
    assert torch.allclose(y, expected, atol=1e-5)
    print("Correctness OK")
