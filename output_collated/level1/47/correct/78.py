# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_141159/code_12.py
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

# --------------------------------------------------------------
#  Optimized CUDA kernel – direct global‑memory accumulation
# --------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdint.h>

// Each thread computes output[b, j] = sum_i input[b, i, j]
__global__ void sum_dim1_opt_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int B,
    const int D1,
    const int D2)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= B * D2) return;

    const int b = tid / D2;          // batch index
    const int j = tid % D2;          // column index in the output

    float sum = 0.0f;

    // Base offset for this batch (size_t to avoid overflow)
    const size_t base = (size_t)b * D1 * D2;

    // Accumulate over the reduction dimension D1
    for (int i = 0; i < D1; ++i) {
        sum += input[base + (size_t)i * D2 + j];
    }

    // Write result
    output[(size_t)b * D2 + j] = sum;
}

// Host wrapper that chooses a reasonable block size (256 threads)
void sum_dim1_opt(torch::Tensor input, torch::Tensor output) {
    const int B = input.size(0);
    const int D1 = input.size(1);
    const int D2 = input.size(2);

    const int threads = 256;
    const int blocks = (B * D2 + threads - 1) / threads;

    sum_dim1_opt_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        B, D1, D2);
}
"""

# --------------------------------------------------------------
#  C++ binding (pybind11)
# --------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void sum_dim1_opt(torch::Tensor input, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sum_dim1_opt", &sum_dim1_opt,
          "Sum along dimension 1 (optimized, no sync)");
}
"""

# --------------------------------------------------------------
#  Compile the inline extension
# --------------------------------------------------------------
sum_ext = load_inline(
    name='sum_dim1_opt',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# --------------------------------------------------------------
#  Functional wrapper required by the evaluator
# --------------------------------------------------------------
def functional_model(x, *, dim):
    """Perform sum over dimension `dim`.
    Only dim==1 is supported."""
    assert dim == 1, "Only reduction over dim=1 is implemented"

    # Ensure a contiguous layout for correct pointer arithmetic
    x = x.contiguous()

    B, D1, D2 = x.shape
    # Allocate output: (B, D2)
    output = torch.empty((B, D2), device=x.device, dtype=x.dtype)

    # Launch the optimized kernel
    sum_ext.sum_dim1_opt(x, output)

    # Return shape (B, 1, D2) to match the original API
    return output.unsqueeze(1)
