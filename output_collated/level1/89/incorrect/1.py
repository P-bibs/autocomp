# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_072553/code_6.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['dim']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['dim']
REQUIRED_FLAT_STATE_NAMES = []


class ModelNew(nn.Module):
    """
    A simple model that performs a cumulative sum (prefix sum) operation along a specified dimension.

    Parameters:
        dim (int): The dimension along which to perform the scan operation.
    """

    def __init__(self, dim):
        """
        Initialize the Scan model.

        Args:
            dim (int): The dimension along which to perform the cumulative sum.
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

# We implement a parallel scan kernel. For row-wise (dim=1) cumsum,
# each row can be processed in parallel.
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void cumsum_dim1_kernel(const float* __restrict__ input, float* __restrict__ output, int N) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    // Each block handles one row of size N=32768
    // Using simple exclusive-to-inclusive transition
    float acc = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        acc += input[row * N + i];
        output[row * N + i] = acc;
    }
}

void cumsum_dim1(torch::Tensor input, torch::Tensor output) {
    const int N = input.size(1);
    const int rows = input.size(0);
    
    // Grid: one block per row, up to device limits
    // For 32768, we use a single block per row
    cumsum_dim1_kernel<<<rows, 1024>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        N
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void cumsum_dim1(torch::Tensor input, torch::Tensor output);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cumsum_dim1", &cumsum_dim1, "Optimized cumsum dim 1");
}
"""

# Compile extension
cumsum_ext = load_inline(
    name='cumsum_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

batch_size = 32768
input_shape = (32768,)
dim = 1

def functional_model(x, *, dim):
    if dim != 1:
        return torch.cumsum(x, dim=dim)
    
    # Pre-allocate output to match input
    output = torch.empty_like(x)
    cumsum_ext.cumsum_dim1(x, output)
    return output

def get_init_inputs():
    return [dim]

def get_inputs():
    return [torch.rand(batch_size, *input_shape, device='cuda')]
