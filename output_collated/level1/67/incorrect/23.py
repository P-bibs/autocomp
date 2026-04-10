# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_161119/code_0.py
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
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused 1D convolution
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

template <typename scalar_t>
__global__ void conv1d_forward_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int input_length,
    const int output_length,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation) {
    
    // Calculate global thread index
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int out_ch = blockIdx.y;
    int batch_idx = blockIdx.z;
    
    // Check bounds
    if (out_idx >= output_length || out_ch >= out_channels || batch_idx >= batch_size) {
        return;
    }
    
    // Calculate output position
    scalar_t sum = 0.0;
    
    // Perform convolution
    for (int in_ch = 0; in_ch < in_channels; ++in_ch) {
        for (int k = 0; k < kernel_size; ++k) {
            int input_pos = out_idx * stride - padding + k * dilation;
            
            if (input_pos >= 0 && input_pos < input_length) {
                int input_idx = batch_idx * (in_channels * input_length) + 
                               in_ch * input_length + input_pos;
                int weight_idx = out_ch * (in_channels * kernel_size) + 
                                in_ch * kernel_size + k;
                
                sum += input[input_idx] * weight[weight_idx];
            }
        }
    }
    
    // Add bias if provided
    if (bias != nullptr) {
        sum += bias[out_ch];
    }
    
    // Write output
    int output_idx = batch_idx * (out_channels * output_length) + 
                     out_ch * output_length + out_idx;
    output[output_idx] = sum;
}

void conv1d_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias,
    torch::Tensor& output,
    const int stride,
    const int padding,
    const int dilation) {
    
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int input_length = input.size(2);
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);
    const int output_length = (input_length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    // Calculate grid and block dimensions
    const int threads_per_block = 256;
    const int blocks_per_grid_x = (output_length + threads_per_block - 1) / threads_per_block;
    const dim3 blocks(blocks_per_grid_x, out_channels, batch_size);
    const dim3 threads(threads_per_block);
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv1d_forward_kernel", ([&] {
        if (bias.has_value()) {
            conv1d_forward_kernel<scalar_t><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(),
                weight.data_ptr<scalar_t>(),
                bias.value().data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch_size,
                in_channels,
                out_channels,
                input_length,
                output_length,
                kernel_size,
                stride,
                padding,
                dilation
            );
        } else {
            conv1d_forward_kernel<scalar_t><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(),
                weight.data_ptr<scalar_t>(),
                nullptr,
                output.data_ptr<scalar_t>(),
                batch_size,
                in_channels,
                out_channels,
                input_length,
                output_length,
                kernel_size,
                stride,
                padding,
                dilation
            );
        }
    }));
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("CUDA kernel launch failed: ", cudaGetErrorString(err));
    }
}
"""

# C++ binding code
cpp_source = r"""
#include <torch/extension.h>

void conv1d_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias,
    torch::Tensor& output,
    const int stride,
    const int padding,
    const int dilation);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv1d_forward", &conv1d_forward, "1D convolution forward pass");
}
"""

# Compile the extension
conv1d_ext = load_inline(
    name='conv1d_fused',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
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
    # Validate groups (only supporting groups=1 for this implementation)
    if conv1d_groups != 1:
        raise NotImplementedError("Only groups=1 is supported in this optimized version")
    
    # Calculate output dimensions
    batch_size = x.size(0)
    out_channels = conv1d_weight.size(0)
    input_length = x.size(2)
    kernel_size = conv1d_weight.size(2)
    
    output_length = (input_length + 2 * conv1d_padding - conv1d_dilation * (kernel_size - 1) - 1) // conv1d_stride + 1
    
    # Create output tensor
    output = torch.empty((batch_size, out_channels, output_length), dtype=x.dtype, device=x.device)
    
    # Call custom CUDA kernel
    conv1d_ext.conv1d_forward(
        x, conv1d_weight, conv1d_bias, output, 
        conv1d_stride, conv1d_padding, conv1d_dilation
    )
    
    return output

batch_size = 32
in_channels = 64
out_channels = 128
kernel_size = 3
length = 131072

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]  # Provide in_channels, out_channels, kernel_size for initialization

def get_inputs():
    x = torch.rand(batch_size, in_channels, length)
    return [x]
