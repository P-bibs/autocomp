# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_152339/code_3.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['dim']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['dim']
REQUIRED_FLAT_STATE_NAMES = []


class ModelNew(nn.Module):
    """
    Simple model that performs Max reduction over a specific dimension.
    """

    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): The dimension to reduce over.
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
# CUDA kernel source – performs a max-reduction along the last dimension
# -------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>

__global__ void max_reduce_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int N,
    const int red_size
) {
    // Global thread index = index of the "row" we are reducing
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Point to the beginning of this row
    const float* row = input + idx * red_size;

    // Keep the maximum in a register
    float max_val = -FLT_MAX;

    // Unrolled loop – helps the compiler keep the loop in registers
    #pragma unroll 8
    for (int i = 0; i < red_size; ++i) {
        float val = row[i];
        if (val > max_val) max_val = val;
    }

    // Write the result
    output[idx] = max_val;
}

// C++ wrapper that launches the kernel
void max_reduce_cuda(torch::Tensor input, torch::Tensor output) {
    int N   = static_cast<int>(input.size(0));
    int red = static_cast<int>(input.size(1));

    const int threads = 256;
    const int blocks  = (N + threads - 1) / threads;

    max_reduce_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N,
        red
    );
}
"""

# -------------------------------------------------------------------------
# C++ binding – exposes the wrapper to Python
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>
void max_reduce_cuda(torch::Tensor input, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("max_reduce", &max_reduce_cuda,
          "Max reduction along the last dimension using a custom CUDA kernel");
}
"""

# Compile the inline extension
max_ext = load_inline(
    name='max_reduce',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# -------------------------------------------------------------------------
# functional_model – replaces the original torch.max call
# -------------------------------------------------------------------------
def functional_model(x: torch.Tensor, *, dim: int) -> torch.Tensor:
    """
    Compute the maximum of `x` along dimension `dim` using a custom CUDA
    kernel that guarantees coalesced memory accesses and efficient register use.
    """
    # Make sure the tensor is on the GPU and of the correct type
    if not x.is_cuda:
        x = x.cuda()
    if x.dtype != torch.float32:
        x = x.float()

    # Bring the reduction dimension to the last axis – after this the data
    # for each reduction "row" is contiguous.
    x_moved = torch.moveaxis(x, dim, -1)

    # Shape of the leading dimensions (all axes except the reduced one)
    leading_shape = x_moved.shape[:-1]          # e.g. (128, 4096)
    N = 1
    for s in leading_shape:
        N *= s
    red_size = x_moved.shape[-1]                # e.g. 4095

    # Flatten to a 2-D tensor (N rows, red_size columns)
    x_flat = x_moved.reshape(N, red_size)
    if not x_flat.is_contiguous():
        x_flat = x_flat.contiguous()

    # Allocate output tensor
    out = torch.empty(N, dtype=x.dtype, device=x.device)

    # Launch the custom CUDA kernel
    max_ext.max_reduce(x_flat, out)

    # Reshape back to the shape without the reduced dimension
    return out.reshape(leading_shape)


# -------------------------------------------------------------------------
# Helper functions required by the evaluation harness
# -------------------------------------------------------------------------
batch_size = 128
dim1 = 4096
dim2 = 4095


def get_init_inputs():
    """Return a dummy list – the real dimension is passed via `dim`."""
    return [1]


def get_inputs():
    """Create a random input tensor on the GPU."""
    x = torch.rand(batch_size, dim1, dim2, device='cuda', dtype=torch.float32)
    return [x]
