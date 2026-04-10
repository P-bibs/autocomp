# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_123708/code_0.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'min_value', 'divisor']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'min_value', 'divisor']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias']


class ModelNew(nn.Module):
    """
    A model that performs a transposed 3D convolution, clamps the output to a minimum value, 
    and then divides the result by a constant.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, min_value, divisor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.min_value = min_value
        self.divisor = divisor

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
    # State for conv_transpose (nn.ConvTranspose3d)
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
    if 'min_value' in flat_state:
        state_kwargs['min_value'] = flat_state['min_value']
    else:
        state_kwargs['min_value'] = getattr(model, 'min_value')
    if 'divisor' in flat_state:
        state_kwargs['divisor'] = flat_state['divisor']
    else:
        state_kwargs['divisor'] = getattr(model, 'divisor')
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

# CUDA kernel that fuses conv_transpose3d + clamp + division
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_conv_transpose3d_clamp_div_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_depth,
    int input_height,
    int input_width,
    int output_depth,
    int output_height,
    int output_width,
    int kernel_depth,
    int kernel_height,
    int kernel_width,
    int stride_d,
    int stride_h,
    int stride_w,
    int padding_d,
    int padding_h,
    int padding_w,
    float min_value,
    float divisor) {
    
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = batch_size * out_channels * output_depth * output_height * output_width;
    
    if (out_idx >= total_outputs) return;
    
    // Calculate output indices
    int w_out = out_idx % output_width;
    int h_out = (out_idx / output_width) % output_height;
    int d_out = (out_idx / (output_width * output_height)) % output_depth;
    int c_out = (out_idx / (output_width * output_height * output_depth)) % out_channels;
    int b = out_idx / (output_width * output_height * output_depth * out_channels);
    
    float sum = 0.0f;
    
    // Convolution transpose operation
    for (int c_in = 0; c_in < in_channels; c_in++) {
        for (int kd = 0; kd < kernel_depth; kd++) {
            for (int kh = 0; kh < kernel_height; kh++) {
                for (int kw = 0; kw < kernel_width; kw++) {
                    // Calculate input position
                    int d_in = d_out - kd + padding_d;
                    int h_in = h_out - kh + padding_h;
                    int w_in = w_out - kw + padding_w;
                    
                    // Check if within input bounds and stride alignment
                    if (d_in >= 0 && d_in < input_depth * stride_d && d_in % stride_d == 0 &&
                        h_in >= 0 && h_in < input_height * stride_h && h_in % stride_h == 0 &&
                        w_in >= 0 && w_in < input_width * stride_w && w_in % stride_w == 0) {
                        
                        d_in /= stride_d;
                        h_in /= stride_h;
                        w_in /= stride_w;
                        
                        if (d_in < input_depth && h_in < input_height && w_in < input_width) {
                            int input_idx = b * (in_channels * input_depth * input_height * input_width) +
                                          c_in * (input_depth * input_height * input_width) +
                                          d_in * (input_height * input_width) +
                                          h_in * input_width +
                                          w_in;
                                          
                            int weight_idx = c_in * (out_channels * kernel_depth * kernel_height * kernel_width) +
                                           c_out * (kernel_depth * kernel_height * kernel_width) +
                                           kd * (kernel_height * kernel_width) +
                                           kh * kernel_width +
                                           kw;
                                           
                            sum += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
    }
    
    // Add bias
    sum += bias[c_out];
    
    // Apply clamp and division
    sum = fmaxf(sum, min_value);
    sum = sum / divisor;
    
    output[out_idx] = sum;
}

void fused_conv_transpose3d_clamp_div_forward(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_depth,
    int input_height,
    int input_width,
    int output_depth,
    int output_height,
    int output_width,
    int kernel_depth,
    int kernel_height,
    int kernel_width,
    int stride_d,
    int stride_h,
    int stride_w,
    int padding_d,
    int padding_h,
    int padding_w,
    float min_value,
    float divisor) {
    
    int total_outputs = batch_size * out_channels * output_depth * output_height * output_width;
    int threads_per_block = 256;
    int blocks = (total_outputs + threads_per_block - 1) / threads_per_block;
    
    fused_conv_transpose3d_clamp_div_kernel<<<blocks, threads_per_block>>>(
        input, weight, bias, output,
        batch_size, in_channels, out_channels,
        input_depth, input_height, input_width,
        output_depth, output_height, output_width,
        kernel_depth, kernel_height, kernel_width,
        stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w,
        min_value, divisor
    );
}
"""

# C++ interface
cpp_source = r"""
#include <torch/extension.h>

void fused_conv_transpose3d_clamp_div_forward(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_depth,
    int input_height,
    int input_width,
    int output_depth,
    int output_height,
    int output_width,
    int kernel_depth,
    int kernel_height,
    int kernel_width,
    int stride_d,
    int stride_h,
    int stride_w,
    int padding_d,
    int padding_h,
    int padding_w,
    float min_value,
    float divisor);

torch::Tensor fused_op(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                      int kernel_depth, int kernel_height, int kernel_width,
                      int stride_d, int stride_h, int stride_w,
                      int padding_d, int padding_h, int padding_w,
                      float min_value, float divisor) {
    // Calculate output dimensions
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_depth = input.size(2);
    int input_height = input.size(3);
    int input_width = input.size(4);
    
    int output_depth = (input_depth - 1) * stride_d - 2 * padding_d + kernel_depth;
    int output_height = (input_height - 1) * stride_h - 2 * padding_h + kernel_height;
    int output_width = (input_width - 1) * stride_w - 2 * padding_w + kernel_width;
    
    // Create output tensor
    auto output = torch::zeros({batch_size, weight.size(0), output_depth, output_height, output_width}, 
                              torch::TensorOptions().dtype(torch::kFloat32).device(input.device()));
    
    // Launch kernel
    fused_conv_transpose3d_clamp_div_forward(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, weight.size(0),
        input_depth, input_height, input_width,
        output_depth, output_height, output_width,
        kernel_depth, kernel_height, kernel_width,
        stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w,
        min_value, divisor
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op, "Fused Conv Transpose 3D + Clamp + Division");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_conv_transpose3d_clamp_div',
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
    min_value,
    divisor,
):
    # Use our fused CUDA kernel instead of separate PyTorch operations
    kernel_depth, kernel_height, kernel_width = conv_transpose_weight.shape[2], conv_transpose_weight.shape[3], conv_transpose_weight.shape[4]
    
    # Handle both scalar and tuple stride/padding
    if isinstance(conv_transpose_stride, int):
        stride_d = stride_h = stride_w = conv_transpose_stride
    else:
        stride_d, stride_h, stride_w = conv_transpose_stride
    
    if isinstance(conv_transpose_padding, int):
        padding_d = padding_h = padding_w = conv_transpose_padding
    else:
        padding_d, padding_h, padding_w = conv_transpose_padding

    return fused_ext.fused_op(
        x, 
        conv_transpose_weight, 
        conv_transpose_bias,
        kernel_depth,
        kernel_height,
        kernel_width,
        stride_d,
        stride_h,
        stride_w,
        padding_d,
        padding_h,
        padding_w,
        min_value,
        divisor
    )

batch_size = 16
in_channels = 64
out_channels = 128
depth, height, width = 24, 48, 48
kernel_size = 3
stride = 2
padding = 1
min_value = -1.0
divisor = 2.0

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, min_value, divisor]

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]
