# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_131936/code_15.py
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
#  CUDA kernel with fully coalesced memory accesses
# --------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void sum_dim1_kernel(const float* __restrict__ input,
                                float* __restrict__ output,
                                int B, int D1, int D2) {
    // batch index
    int b = blockIdx.x;
    // column (output) index – one thread per column
    int j = blockIdx.y * blockDim.x + threadIdx.x;

    if (b >= B || j >= D2) return;

    float sum = 0.0f;

    // Unroll the inner loop for better instruction throughput
    #pragma unroll 8
    for (int i = 0; i < D1; ++i) {
        // Each warp accesses consecutive j's for a given i → coalesced
        sum += input[b * D1 * D2 + i * D2 + j];
    }

    output[b * D2 + j] = sum;
}

void sum_dim1(torch::Tensor input, torch::Tensor output) {
    int B = input.size(0);
    int D1 = input.size(1);
    int D2 = input.size(2);

    const int threads = 256;                     // multiple of 32 → max occupancy
    dim3 blocks(B, (D2 + threads - 1) / threads); // grid covers all (b, j) pairs

    sum_dim1_kernel<<<blocks, threads>>>(
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

void sum_dim1(torch::Tensor input, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sum_dim1", &sum_dim1, "Sum along dim=1 with coalesced memory accesses");
}
"""

# --------------------------------------------------------------
#  Compile the extension
# --------------------------------------------------------------
sum_ext = load_inline(
    name='sum_dim1_coalesced',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# --------------------------------------------------------------
#  Functional model required by the evaluation harness
# --------------------------------------------------------------
def functional_model(x, *, dim):
    """Sum tensor `x` over dimension `dim` (only dim==1 supported)."""
    assert dim == 1
    # Output shape: (batch_size, 1, dim2)
    output = torch.empty((x.shape[0], x.shape[2]), device=x.device, dtype=x.dtype)
    sum_ext.sum_dim1(x, output)
    return output.unsqueeze(1)

# --------------------------------------------------------------
#  Helper functions for the evaluation script (not part of the model)
# --------------------------------------------------------------
batch_size = 128
dim1 = 4096
dim2 = 4095

def get_init_inputs():
    return [1]                     # dim = 1

def get_inputs():
    return [torch.rand(batch_size, dim1, dim2, device='cuda')]
