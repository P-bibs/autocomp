# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_120759/code_0.py
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

# Define the fused CUDA kernel
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_conv_transpose3d_clamp_div_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_depth,
    int input_height,
    int input_width,
    int output_depth,
    int output_height,
    int output_width,
    int kernel_size,
    int stride,
    int padding,
    int output_padding_d,
    int output_padding_h,
    int output_padding_w,
    float min_value,
    float divisor) {
    
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output_elements = batch_size * out_channels * output_depth * output_height * output_width;
    
    if (out_idx >= total_output_elements) return;
    
    // Calculate output indices
    int temp = out_idx;
    int w_out = temp % output_width;
    temp /= output_width;
    int h_out = temp % output_height;
    temp /= output_height;
    int d_out = temp % output_depth;
    temp /= output_depth;
    int c_out = temp % out_channels;
    int n = temp / out_channels;
    
    float result = 0.0f;
    
    if (bias != nullptr) {
        result = bias[c_out];
    }
    
    // Convolution transpose computation
    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int kd = 0; kd < kernel_size; ++kd) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    // Calculate corresponding input position
                    int d_in = d_out + padding - kd;
                    int h_in = h_out + padding - kh;
                    int w_in = w_out + padding - kw;
                    
                    // Check if within input bounds after accounting for stride
                    if (d_in >= 0 && d_in < input_depth * stride && d_in % stride == 0 &&
                        h_in >= 0 && h_in < input_height * stride && h_in % stride == 0 &&
                        w_in >= 0 && w_in < input_width * stride && w_in % stride == 0) {
                        
                        int d_in_mapped = d_in / stride;
                        int h_in_mapped = h_in / stride;
                        int w_in_mapped = w_in / stride;
                        
                        if (d_in_mapped < input_depth && h_in_mapped < input_height && w_in_mapped < input_width) {
                            int input_idx = n * (in_channels * input_depth * input_height * input_width) +
                                          c_in * (input_depth * input_height * input_width) +
                                          d_in_mapped * (input_height * input_width) +
                                          h_in_mapped * input_width +
                                          w_in_mapped;
                                          
                            int weight_idx = c_in * (out_channels * kernel_size * kernel_size * kernel_size) +
                                           c_out * (kernel_size * kernel_size * kernel_size) +
                                           kd * (kernel_size * kernel_size) +
                                           kh * kernel_size +
                                           kw;
                            
                            result += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
    }
    
    // Apply clamp and division
    result = fmaxf(min_value, result);
    result = result / divisor;
    
    output[out_idx] = result;
}

void fused_conv_transpose3d_clamp_div_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    int kernel_size,
    int stride,
    int padding,
    int output_padding_d,
    int output_padding_h,
    int output_padding_w,
    float min_value,
    float divisor) {
    
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_depth = input.size(2);
    int input_height = input.size(3);
    int input_width = input.size(4);
    
    int output_depth = (input_depth - 1) * stride - 2 * padding + kernel_size + output_padding_d;
    int output_height = (input_height - 1) * stride - 2 * padding + kernel_size + output_padding_h;
    int output_width = (input_width - 1) * stride - 2 * padding + kernel_size + output_padding_w;
    
    int total_output_elements = batch_size * weight.size(1) * output_depth * output_height * output_width;
    
    const int threads_per_block = 256;
    const int blocks = (total_output_elements + threads_per_block - 1) / threads_per_block;
    
    fused_conv_transpose3d_clamp_div_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        weight.size(1), // out_channels for ConvTranspose3d
        input_depth,
        input_height,
        input_width,
        output_depth,
        output_height,
        output_width,
        kernel_size,
        stride,
        padding,
        output_padding_d,
        output_padding_h,
        output_padding_w,
        min_value,
        divisor
    );
}
"""

# Define the C++ binding
cpp_source = r"""
#include <torch/extension.h>

void fused_conv_transpose3d_clamp_div_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    int kernel_size,
    int stride,
    int padding,
    int output_padding_d,
    int output_padding_h,
    int output_padding_w,
    float min_value,
    float divisor);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_conv_transpose3d_clamp_div_forward, "Fused ConvTranspose3D + Clamp + Div");
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
    # Validate that we only support the default values for unsupported parameters
    if conv_transpose_groups != 1:
        raise NotImplementedError("Groups != 1 is not supported")
    if conv_transpose_dilation != (1, 1, 1):
        raise NotImplementedError("Dilation != (1,1,1) is not supported")
    
    # Extract parameters
    kernel_size = conv_transpose_weight.size(2)  # assuming cubic kernel
    stride = conv_transpose_stride[0] if isinstance(conv_transpose_stride, (list, tuple)) else conv_transpose_stride
    padding = conv_transpose_padding[0] if isinstance(conv_transpose_padding, (list, tuple)) else conv_transpose_padding
    output_padding_d = conv_transpose_output_padding[0] if isinstance(conv_transpose_output_padding, (list, tuple)) else conv_transpose_output_padding
    output_padding_h = conv_transpose_output_padding[1] if isinstance(conv_transpose_output_padding, (list, tuple)) else conv_transpose_output_padding
    output_padding_w = conv_transpose_output_padding[2] if isinstance(conv_transpose_output_padding, (list, tuple)) else conv_transpose_output_padding
    
    # Calculate output dimensions
    out_channels = conv_transpose_weight.size(1)  # For ConvTranspose: out_channels = weight.size(1)
    output_depth = (x.size(2) - 1) * stride - 2 * padding + kernel_size + output_padding_d
    output_height = (x.size(3) - 1) * stride - 2 * padding + kernel_size + output_padding_h
    output_width = (x.size(4) - 1) * stride - 2 * padding + kernel_size + output_padding_w
    
    # Create output tensor
    output = torch.empty(x.size(0), out_channels, output_depth, output_height, output_width, 
                        dtype=x.dtype, device=x.device)
    
    # Call the fused operation
    fused_ext.fused_op(
        x, 
        conv_transpose_weight, 
        conv_transpose_bias,
        output,
        kernel_size,
        stride,
        padding,
        output_padding_d,
        output_padding_h,
        output_padding_w,
        min_value,
        divisor
    )
    
    return output

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
