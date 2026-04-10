# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_160042/code_1.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'dilation', 'groups', 'bias']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv1d_weight', 'conv1d_bias', 'conv1d_stride', 'conv1d_padding', 'conv1d_dilation', 'conv1d_groups']
REQUIRED_FLAT_STATE_NAMES = ['conv1d_weight', 'conv1d_bias']


class ModelNew(nn.Module):
    """
    Performs a standard 1D convolution operation.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int=1, padding: int=0, dilation: int=1, groups: int=1, bias: bool=False):
        super(ModelNew, self).__init__()
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

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
    # State for conv1d (nn.Conv1d)
    if 'conv1d_weight' in flat_state:
        state_kwargs['conv1d_weight'] = flat_state['conv1d_weight']
    else:
        state_kwargs['conv1d_weight'] = getattr(model.conv1d, 'weight', None)
    if 'conv1d_bias' in flat_state:
        state_kwargs['conv1d_bias'] = flat_state['conv1d_bias']
    else:
        state_kwargs['conv1d_bias'] = getattr(model.conv1d, 'bias', None)
    state_kwargs['conv1d_stride'] = model.conv1d.stride
    state_kwargs['conv1d_padding'] = model.conv1d.padding
    state_kwargs['conv1d_dilation'] = model.conv1d.dilation
    state_kwargs['conv1d_groups'] = model.conv1d.groups
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

# CUDA kernel code implementing 1D convolution using implicit GEMM approach
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv1d_implicit_gemm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int input_length,
    int out_channels,
    int kernel_size,
    int output_length
) {
    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check bounds
    if (idx >= batch_size * out_channels * output_length) return;
    
    // Decompose linear index into 3D coordinates (batch, out_channel, output_position)
    int tmp = idx;
    int out_pos = tmp % output_length;
    tmp /= output_length;
    int out_ch = tmp % out_channels;
    int batch = tmp / out_channels;
    
    // Compute convolution at this position
    float result = bias[out_ch];
    
    for (int in_ch = 0; in_ch < in_channels; ++in_ch) {
        for (int k = 0; k < kernel_size; ++k) {
            // Input access pattern: [batch][in_ch][out_pos + k]
            int input_idx = batch * (in_channels * input_length) + 
                           in_ch * input_length + 
                           (out_pos + k);
                           
            // Weight access pattern: [out_ch][in_ch][k]
            int weight_idx = out_ch * (in_channels * kernel_size) + 
                            in_ch * kernel_size + 
                            k;
                            
            result += input[input_idx] * weight[weight_idx];
        }
    }
    
    output[idx] = result;
}

// Host function to launch the kernel
void conv1d_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output) {
    // Extract dimensions
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_length = input.size(2);
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);
    int output_length = input_length - kernel_size + 1;
    
    // Calculate total number of output elements
    int total_elements = batch_size * out_channels * output_length;
    
    // Configure kernel launch parameters
    const int threads_per_block = 256;
    const int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    // Launch kernel
    conv1d_implicit_gemm_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        input_length,
        out_channels,
        kernel_size,
        output_length
    );
}
"""

# C++ binding code
cpp_source = r"""
#include <torch/extension.h>

// Forward declaration of the CUDA function
void conv1d_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output);

// Pybind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv1d_forward", &conv1d_forward, "1D Convolution using Implicit GEMM");
}
"""

# Compile the extension
conv1d_ext = load_inline(
    name='conv1d_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(
    x,
    *,
    conv1d_weight,
    conv1d_bias,
    conv1d_stride,
    conv1d_padding,
    conv1d_dilation,
    conv1d_groups,
):
    # Validate constraints for optimization (basic configuration only)
    assert conv1d_stride == 1, "Only stride=1 is supported"
    assert conv1d_padding == 0, "Only padding=0 is supported"
    assert conv1d_dilation == 1, "Only dilation=1 is supported"
    assert conv1d_groups == 1, "Only groups=1 is supported"
    
    # Get tensor dimensions
    batch_size, in_channels, input_length = x.shape
    out_channels, _, kernel_size = conv1d_weight.shape
    output_length = input_length - kernel_size + 1
    
    # Create output tensor
    output = torch.empty((batch_size, out_channels, output_length), device=x.device, dtype=x.dtype)
    
    # Call custom CUDA implementation
    conv1d_ext.conv1d_forward(x, conv1d_weight, conv1d_bias, output)
    
    return output

# Constants for testing
batch_size = 32
in_channels = 64
out_channels = 128
kernel_size = 3
length = 131072

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]

def get_inputs():
    x = torch.rand(batch_size, in_channels, length)
    return [x]
