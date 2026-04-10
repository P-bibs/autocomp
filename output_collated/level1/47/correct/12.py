# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_123707/code_3.py
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

# Optimization: Coalesce global memory accesses.
# We map the threads to (batch, dim2) to ensure that the innermost dimension 
# (dim2) is traversed contiguously by the threads in a warp.

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void sum_dim1_kernel(const float* __restrict__ input, float* __restrict__ output, 
                                 int B, int D1, int D2) {
    int b = blockIdx.x;
    int d2 = blockIdx.y * blockDim.x + threadIdx.x;

    if (d2 < D2) {
        float sum = 0.0f;
        for (int d1 = 0; d1 < D1; ++d1) {
            sum += input[b * (D1 * D2) + d1 * D2 + d2];
        }
        output[b * D2 + d2] = sum;
    }
}

void sum_dim1_forward(torch::Tensor input, torch::Tensor output) {
    int B = input.size(0);
    int D1 = input.size(1);
    int D2 = input.size(2);

    dim3 blocks(B, (D2 + 31) / 32);
    dim3 threads(32);

    sum_dim1_kernel<<<blocks, threads>>>(input.data_ptr<float>(), output.data_ptr<float>(), B, D1, D2);
}
"""

cpp_source = r"""
#include <torch/extension.h>

void sum_dim1_forward(torch::Tensor input, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sum_dim1_forward", &sum_dim1_forward, "Sum along dimension 1 with coalesced memory access");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='sum_dim1',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, dim):
    # Ensure input is float32 for simplicity
    x = x.float()
    shape = list(x.shape)
    shape[dim] = 1
    output = torch.empty(shape, device=x.device, dtype=x.dtype)
    
    # We specifically handle the reducer for dim=1 as per the problem description
    if dim == 1:
        fused_ext.sum_dim1_forward(x, output)
    else:
        # Generic fallback if dim is not 1 (not requested as primary target)
        return torch.sum(x, dim=dim, keepdim=True)
        
    return output

# Verification/Setup
batch_size = 128
dim1 = 4096
dim2 = 4095
reduce_dim = 1

def get_init_inputs():
    return [reduce_dim]

def get_inputs():
    x = torch.rand(batch_size, dim1, dim2, device='cuda')
    return [x]
