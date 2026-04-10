# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_072553/code_0.py
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# CUDA kernel implementation for cumulative sum
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void cumsum_kernel(
    const float* input,
    float* output,
    const int batch_size,
    const int input_size,
    const int dim
) {
    int batch_idx = blockIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    // Process along dimension 1 (second dimension)
    if (dim == 1) {
        const float* input_row = input + batch_idx * input_size;
        float* output_row = output + batch_idx * input_size;
        
        // Sequential cumsum for each row - most efficient for large rows
        if (threadIdx.x == 0) {
            float sum = 0.0f;
            for (int i = 0; i < input_size; i++) {
                sum += input_row[i];
                output_row[i] = sum;
            }
        }
    }
}

void cumsum_forward(
    const at::Tensor& input,
    at::Tensor& output,
    const int dim
) {
    const int batch_size = input.size(0);
    const int input_size = input.size(1);
    
    const dim3 threads(1);
    const dim3 blocks(batch_size);
    
    cumsum_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        input_size,
        dim
    );
}
"""

# C++ interface/bindings
cpp_source = r"""
#include <torch/extension.h>

void cumsum_forward(
    const at::Tensor& input,
    at::Tensor& output,
    const int dim
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cumsum_cuda", &cumsum_forward, "Cumulative sum CUDA implementation");
}
"""

# Compile the extension
cumsum_ext = load_inline(
    name='cumsum_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(
    x,
    *,
    dim,
):
    # Ensure tensor is on GPU
    if not x.is_cuda:
        x = x.cuda()
    
    # Create output tensor
    output = torch.empty_like(x)
    
    # Call custom CUDA kernel
    cumsum_ext.cumsum_cuda(x, output, dim)
    
    return output

batch_size = 32768
input_shape = (32768,)
dim = 1

def get_init_inputs():
    """
    Returns the initialization parameters for the Scan model.

    Returns:
        list: A list containing the `dim` parameter for model initialization.
    """
    return [dim]

def get_inputs():
    """
    Generates random inputs for testing the Scan model.

    Returns:
        list: A list containing a single randomly generated tensor with shape 
              (batch_size, *input_shape).
    """
    return [torch.rand(batch_size, *input_shape, device='cuda')]
