# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_123254/code_3.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'bias_shape', 'stride', 'padding', 'output_padding']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'bias']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a transposed convolution, subtracts a bias term, and applies tanh activation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, bias_shape, stride=2, padding=1, output_padding=1):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))

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
    # State for conv_transpose (nn.ConvTranspose2d)
    if 'conv_transpose_weight' in flat_state:
        state_kwargs['conv_transpose_weight'] = flat_state['conv_transpose_weight']
    else:
        state_kwargs['conv_transpose_weight'] = getattr(model.conv_transpose, 'weight', None)
    if 'conv_transpose_bias' in flat_state:
        state_kwargs['conv_transpose_bias'] = flat_state['conv_transpose_bias']
    else:
        state_kwargs['conv_transpose_bias'] = getattr(model.conv_transpose, 'bias', None)
    state_kwargs['conv_transpose_stride'] = model.conv_transpose.stride
    state_kwargs['conv_transpose_padding'] = model.conv_transpose.padding
    state_kwargs['conv_transpose_output_padding'] = model.conv_transpose.output_padding
    state_kwargs['conv_transpose_groups'] = model.conv_transpose.groups
    state_kwargs['conv_transpose_dilation'] = model.conv_transpose.dilation
    if 'bias' in flat_state:
        state_kwargs['bias'] = flat_state['bias']
    else:
        state_kwargs['bias'] = getattr(model, 'bias')
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

# CUDA kernel for fused conv transpose + bias subtraction + tanh
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_conv_transpose_tanh_kernel(
    const float* input,
    const float* weight,
    const float* conv_bias,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_height,
    int input_width,
    int output_height,
    int output_width,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    int groups,
    int dilation
) {
    int batch_idx = blockIdx.x;
    int out_ch = blockIdx.y;
    int out_y = (blockIdx.z * blockDim.x + threadIdx.x) / output_width;
    int out_x = (blockIdx.z * blockDim.x + threadIdx.x) % output_width;
    
    if (batch_idx >= batch_size || out_ch >= out_channels || out_y >= output_height || out_x >= output_width) {
        return;
    }
    
    float sum = 0.0f;
    
    // Calculate convolution transpose
    for (int in_ch = 0; in_ch < in_channels; in_ch++) {
        for (int ky = 0; ky < kernel_size; ky++) {
            for (int kx = 0; kx < kernel_size; kx++) {
                // Map output coordinates to input coordinates
                int in_y = (out_y + padding - ky * dilation) / stride;
                int in_x = (out_x + padding - kx * dilation) / stride;
                
                // Check if the input coordinate is valid and aligns with stride
                if ((out_y + padding - ky * dilation) % stride == 0 && 
                    (out_x + padding - kx * dilation) % stride == 0 &&
                    in_y >= 0 && in_y < input_height &&
                    in_x >= 0 && in_x < input_width) {
                    
                    int input_idx = batch_idx * (in_channels * input_height * input_width) +
                                   in_ch * (input_height * input_width) +
                                   in_y * input_width + in_x;
                                   
                    int weight_idx = out_ch * (in_channels * kernel_size * kernel_size) +
                                    in_ch * (kernel_size * kernel_size) +
                                    (kernel_size - 1 - ky) * kernel_size + (kernel_size - 1 - kx);
                    
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    // Add convolution bias if provided
    if (conv_bias != nullptr) {
        sum += conv_bias[out_ch];
    }
    
    // Subtract bias and apply tanh
    sum -= bias[out_ch];
    float result = tanhf(sum);
    
    int output_idx = batch_idx * (out_channels * output_height * output_width) +
                    out_ch * (output_height * output_width) +
                    out_y * output_width + out_x;
    output[output_idx] = result;
}

void fused_conv_transpose_tanh_forward(
    const torch::Tensor input,
    const torch::Tensor weight,
    const torch::Tensor conv_bias,
    const torch::Tensor bias,
    torch::Tensor output,
    int stride,
    int padding,
    int output_padding,
    int groups,
    int dilation
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_height = input.size(2);
    int input_width = input.size(3);
    int out_channels = output.size(1);
    int output_height = output.size(2);
    int output_width = output.size(3);
    int kernel_size = weight.size(2);
    
    // Grid and block dimensions
    int threads_per_block = 256;
    int total_output_elements = output_height * output_width;
    int blocks_per_sample = (total_output_elements + threads_per_block - 1) / threads_per_block;
    
    dim3 grid(batch_size, out_channels, blocks_per_sample);
    dim3 block(threads_per_block);
    
    fused_conv_transpose_tanh_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        conv_bias.defined() ? conv_bias.data_ptr<float>() : nullptr,
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_height,
        input_width,
        output_height,
        output_width,
        kernel_size,
        stride,
        padding,
        output_padding,
        groups,
        dilation
    );
    
    cudaDeviceSynchronize();
}
"""

# C++ interface
cpp_source = r"""
#include <torch/extension.h>

void fused_conv_transpose_tanh_forward(
    const torch::Tensor input,
    const torch::Tensor weight,
    const torch::Tensor conv_bias,
    const torch::Tensor bias,
    torch::Tensor output,
    int stride,
    int padding,
    int output_padding,
    int groups,
    int dilation
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_transpose_tanh", &fused_conv_transpose_tanh_forward, "Fused conv transpose with tanh");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_conv_transpose_tanh_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(
    x,
    *,
    conv_transpose_weight,
    conv_transpose_bias,
    conv_transpose_stride,
    conv_transpose_padding,
    conv_transpose_output_padding,
    conv_transpose_groups,
    conv_transpose_dilation,
    bias,
):
    # Calculate output dimensions
    batch_size, in_channels, input_height, input_width = x.shape
    out_channels, _, kernel_size, _ = conv_transpose_weight.shape
    
    # Calculate output dimensions for conv transpose
    output_height = (input_height - 1) * conv_transpose_stride + (kernel_size - 1) * conv_transpose_dilation + 1 - 2 * conv_transpose_padding + conv_transpose_output_padding
    output_width = (input_width - 1) * conv_transpose_stride + (kernel_size - 1) * conv_transpose_dilation + 1 - 2 * conv_transpose_padding + conv_transpose_output_padding
    
    # Create output tensor
    output = torch.empty((batch_size, out_channels, output_height, output_width), device=x.device, dtype=x.dtype)
    
    # Call fused kernel
    fused_ext.fused_conv_transpose_tanh(
        x, 
        conv_transpose_weight, 
        conv_transpose_bias if conv_transpose_bias is not None else torch.empty(0, device=x.device, dtype=x.dtype),
        bias,
        output,
        conv_transpose_stride,
        conv_transpose_padding,
        conv_transpose_output_padding,
        conv_transpose_groups,
        conv_transpose_dilation
    )
    
    return output

# Test parameters
batch_size = 32
in_channels = 64  
out_channels = 64  
height = width = 256 
kernel_size = 4
bias_shape = (out_channels, 1, 1)

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]
