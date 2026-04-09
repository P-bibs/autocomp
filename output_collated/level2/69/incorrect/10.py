# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_050905/code_0.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_weight', 'conv_bias', 'conv_stride', 'conv_padding', 'conv_dilation', 'conv_groups']
REQUIRED_FLAT_STATE_NAMES = ['conv_weight', 'conv_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a convolution, applies HardSwish, and then ReLU.
    """

    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

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

# CUDA kernel that fuses convolution + hardswish + relu
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

__global__ void fused_conv_activation_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    int groups) {
    
    int out_height = (height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_width = (width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    int total_threads = batch_size * out_channels * out_height * out_width;
    CUDA_1D_KERNEL_LOOP(index, total_threads) {
        int tmp = index;
        int w_out = tmp % out_width;
        tmp /= out_width;
        int h_out = tmp % out_height;
        tmp /= out_height;
        int c_out = tmp % out_channels;
        int b = tmp / out_channels;
        
        float sum = 0.0f;
        
        // Calculate convolution
        for (int c_in = 0; c_in < in_channels; c_in++) {
            for (int kh = 0; kh < kernel_size; kh++) {
                for (int kw = 0; kw < kernel_size; kw++) {
                    int h_in = h_out * stride - padding + kh * dilation;
                    int w_in = w_out * stride - padding + kw * dilation;
                    
                    if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                        int input_idx = b * (in_channels * height * width) + 
                                        c_in * (height * width) + 
                                        h_in * width + w_in;
                        int weight_idx = c_out * (in_channels * kernel_size * kernel_size) + 
                                         c_in * (kernel_size * kernel_size) + 
                                         kh * kernel_size + kw;
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        
        // Add bias
        sum += bias[c_out];
        
        // Apply hardswish: x * relu6(x + 3) / 6
        float hardswish_val;
        if (sum <= -3.0f) {
            hardswish_val = 0.0f;
        } else if (sum >= 3.0f) {
            hardswish_val = sum;
        } else {
            hardswish_val = sum * (sum + 3.0f) / 6.0f;
        }
        
        // Apply relu
        float relu_val = fmaxf(hardswish_val, 0.0f);
        
        output[index] = relu_val;
    }
}

void fused_conv_activation_forward(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    int groups) {
    
    int out_height = (height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_width = (width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int total_elements = batch_size * out_channels * out_height * out_width;
    
    const int threads_per_block = 256;
    const int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    fused_conv_activation_kernel<<<blocks, threads_per_block>>>(
        input, weight, bias, output,
        batch_size, in_channels, out_channels, height, width,
        kernel_size, stride, padding, dilation, groups
    );
}
"""

# C++ interface/bindings
cpp_source = r"""
#include <torch/extension.h>

void fused_conv_activation_forward(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    int groups);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_activation", &fused_conv_activation_forward, "Fused Convolution + Hardswish + ReLU");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_conv_activation_ext',
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
):
    batch_size, in_channels, height, width = x.shape
    out_channels = conv_weight.shape[0]
    kernel_size = conv_weight.shape[2]
    
    # Calculate output dimensions
    out_height = (height + 2 * conv_padding - conv_dilation * (kernel_size - 1) - 1) // conv_stride + 1
    out_width = (width + 2 * conv_padding - conv_dilation * (kernel_size - 1) - 1) // conv_stride + 1
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, out_height, out_width, device=x.device, dtype=x.dtype)
    
    # Call fused kernel
    fused_ext.fused_conv_activation(
        x.data_ptr(),
        conv_weight.data_ptr(),
        conv_bias.data_ptr(),
        output.data_ptr(),
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        kernel_size,
        conv_stride,
        conv_padding,
        conv_dilation,
        conv_groups
    )
    
    return output

batch_size = 128
in_channels = 8
out_channels = 64
height, width = 128, 128
kernel_size = 3

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]
