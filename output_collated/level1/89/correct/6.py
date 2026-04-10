# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_073025/code_6.py
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

# CUDA kernel: optimized for row-major cumsum
# Given tensor of (batch_size, dim_size), we parallelize across batch_size (rows)
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

template<typename scalar_t>
__global__ void cumsum_row_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int batch_size,
    int dim_size
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= batch_size) return;

    const scalar_t* row_in = input + row * dim_size;
    scalar_t* row_out = output + row * dim_size;

    scalar_t running_sum = 0;
    for (int col = 0; col < dim_size; ++col) {
        running_sum += row_in[col];
        row_out[col] = running_sum;
    }
}

void cumsum_forward(torch::Tensor input, torch::Tensor output) {
    int batch_size = input.size(0);
    int dim_size = input.size(1);
    
    const int threads = 256;
    const int blocks = (batch_size + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "cumsum_row_kernel", ([&] {
        cumsum_row_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            dim_size
        );
    }));
}
"""

cpp_source = r"""
#include <torch/extension.h>
void cumsum_forward(torch::Tensor input, torch::Tensor output);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cumsum_fused", &cumsum_forward, "Fused row-wise cumsum");
}
"""

# Compile the extension
cumsum_ext = load_inline(
    name='cumsum_fused',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, dim):
    """
    Optimized cumsum along the specified dimension.
    For (batch_size, dim_size), this kernel processes each row in parallel.
    """
    # Force output to be on GPU and same dtype
    output = torch.empty_like(x)
    
    # Simple check for dim=1 logic
    if dim == 1:
        cumsum_ext.cumsum_fused(x, output)
    else:
        # Fallback to standard if dim is not 1, 
        # though plan implies processing (32768, 32768)
        output = torch.cumsum(x, dim=dim)
        
    return output

batch_size = 32768
input_shape = (32768,)
dim = 1

def get_init_inputs():
    return [dim]

def get_inputs():
    return [torch.rand(batch_size, *input_shape, device='cuda')]
