# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_145843/code_1.py
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

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <climits>

__global__ void max_kernel(const float* __restrict__ input, float* __restrict__ output, 
                           int batch_size, int dim1, int dim2) {
    // Each thread handles one output element (batch_id, dim2_id)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * dim2) return;

    int batch_id = idx / dim2;
    int dim2_id = idx % dim2;

    float max_val = -1e38f; // Sufficiently small initial value for float
    const float* row_start = input + batch_id * dim1 * dim2 + dim2_id;
    
    // Traverse dim1 (reduction dimension) with stride dim2 for coalesced reads within warps
    for (int i = 0; i < dim1; ++i) {
        float val = row_start[i * dim2];
        if (val > max_val) max_val = val;
    }
    
    output[idx] = max_val;
}

void max_forward(torch::Tensor input, torch::Tensor output) {
    auto batch_size = input.size(0);
    auto dim1 = input.size(1);
    auto dim2 = input.size(2);
    
    const int threads = 256;
    const int blocks = (batch_size * dim2 + threads - 1) / threads;
    
    max_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        batch_size, dim1, dim2
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void max_forward(torch::Tensor input, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("max_forward", &max_forward, "Custom Max Reduction Forward");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='max_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, dim):
    assert dim == 1, "Optimized kernel only supports reduction over dimension 1"
    batch_size, dim1, dim2 = x.shape
    output = torch.empty((batch_size, dim2), device=x.device, dtype=x.dtype)
    fused_ext.max_forward(x.contiguous(), output)
    return output

# Setup for testing
batch_size = 128
dim1 = 4096
dim2 = 4095

def get_inputs():
    x = torch.rand(batch_size, dim1, dim2, device='cuda')
    return [x]
