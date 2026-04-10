# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_141042/code_0.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'subtract_value_1', 'subtract_value_2']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_weight', 'conv_bias', 'conv_stride', 'conv_padding', 'conv_dilation', 'conv_groups', 'subtract_value_1', 'subtract_value_2']
REQUIRED_FLAT_STATE_NAMES = ['conv_weight', 'conv_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a convolution, subtracts two values, applies Mish activation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value_1 = subtract_value_1
        self.subtract_value_2 = subtract_value_2

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
    # State for conv (nn.Conv2d)
    if 'conv_weight' in flat_state:
        state_kwargs['conv_weight'] = flat_state['conv_weight']
    else:
        state_kwargs['conv_weight'] = getattr(model.conv, 'weight', None)
    if 'conv_bias' in flat_state:
        state_kwargs['conv_bias'] = flat_state['conv_bias']
    else:
        state_kwargs['conv_bias'] = getattr(model.conv, 'bias', None)
    state_kwargs['conv_stride'] = model.conv.stride
    state_kwargs['conv_padding'] = model.conv.padding
    state_kwargs['conv_dilation'] = model.conv.dilation
    state_kwargs['conv_groups'] = model.conv.groups
    if 'subtract_value_1' in flat_state:
        state_kwargs['subtract_value_1'] = flat_state['subtract_value_1']
    else:
        state_kwargs['subtract_value_1'] = getattr(model, 'subtract_value_1')
    if 'subtract_value_2' in flat_state:
        state_kwargs['subtract_value_2'] = flat_state['subtract_value_2']
    else:
        state_kwargs['subtract_value_2'] = getattr(model, 'subtract_value_2')
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

# CUDA kernel for fused conv + subtract + mish operation
from torch.utils.cpp_extension import load_inline

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__device__ float mish_activation(float x) {
    return x * tanhf(logf(1.0f + expf(x)));
}

__global__ void fused_conv_subtract_mish_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    float subtract_value_1,
    float subtract_value_2) {
    
    // Calculate output dimensions
    int out_height = (height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_width = (width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    // Global thread indices
    int batch_idx = blockIdx.x;
    int out_ch = blockIdx.y;
    int thread_idx = threadIdx.x;
    
    if (batch_idx >= batch_size || out_ch >= out_channels) return;
    
    // Load bias for this output channel
    float bias_val = bias[out_ch];
    
    // Process output pixels
    for (int out_y = thread_idx; out_y < out_height; out_y += blockDim.x) {
        for (int out_x = 0; out_x < out_width; out_x++) {
            float conv_sum = bias_val;
            
            // Perform convolution
            for (int in_ch = 0; in_ch < in_channels; in_ch++) {
                for (int ky = 0; ky < kernel_size; ky++) {
                    for (int kx = 0; kx < kernel_size; kx++) {
                        int in_y = out_y * stride - padding + ky * dilation;
                        int in_x = out_x * stride - padding + kx * dilation;
                        
                        if (in_y >= 0 && in_y < height && in_x >= 0 && in_x < width) {
                            int input_idx = batch_idx * (in_channels * height * width) + 
                                          in_ch * (height * width) + 
                                          in_y * width + in_x;
                            int weight_idx = out_ch * (in_channels * kernel_size * kernel_size) + 
                                           in_ch * (kernel_size * kernel_size) + 
                                           ky * kernel_size + kx;
                            
                            conv_sum += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
            
            // Apply subtractions and mish activation
            float result = conv_sum - subtract_value_1 - subtract_value_2;
            result = mish_activation(result);
            
            // Write to output
            int output_idx = batch_idx * (out_channels * out_height * out_width) + 
                           out_ch * (out_height * out_width) + 
                           out_y * out_width + out_x;
            output[output_idx] = result;
        }
    }
}

void fused_conv_subtract_mish_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    float subtract_value_1,
    float subtract_value_2) {
    
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    int out_channels = weight.size(0);
    
    int out_height = (height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_width = (width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    // Launch configuration
    dim3 grid(batch_size, out_channels);
    dim3 block(min(512, out_width * out_height));
    
    fused_conv_subtract_mish_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        kernel_size,
        stride,
        padding,
        dilation,
        subtract_value_1,
        subtract_value_2
    );
}
"""

# C++ binding code
cpp_source = r"""
#include <torch/extension.h>

void fused_conv_subtract_mish_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    float subtract_value_1,
    float subtract_value_2);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_subtract_mish", &fused_conv_subtract_mish_forward, "Fused Conv2D + Subtract + Mish");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_conv_subtract_mish',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(
    x,
    *,
    conv_weight,
    conv_bias,
    conv_stride,
    conv_padding,
    conv_dilation,
    conv_groups,
    subtract_value_1,
    subtract_value_2,
):
    # Ensure groups is 1 for this implementation
    if conv_groups != 1:
        raise NotImplementedError("Grouped convolutions not supported in this fused implementation")
    
    # Create output tensor with correct shape
    out_channels = conv_weight.size(0)
    in_height, in_width = x.shape[2], x.shape[3]
    kernel_size = conv_weight.size(2)
    
    out_height = (in_height + 2 * conv_padding - conv_dilation * (kernel_size - 1) - 1) // conv_stride + 1
    out_width = (in_width + 2 * conv_padding - conv_dilation * (kernel_size - 1) - 1) // conv_stride + 1
    
    output = torch.empty(x.size(0), out_channels, out_height, out_width, device=x.device, dtype=x.dtype)
    
    # Call fused CUDA kernel
    fused_ext.fused_conv_subtract_mish(
        x, conv_weight, conv_bias, output,
        kernel_size, conv_stride, conv_padding, conv_dilation,
        subtract_value_1, subtract_value_2
    )
    
    return output

# Test parameters
batch_size = 128
in_channels = 8
out_channels = 64
height, width = 256, 256
kernel_size = 3
subtract_value_1 = 0.5
subtract_value_2 = 0.2

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]
