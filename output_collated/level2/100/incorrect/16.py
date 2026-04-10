# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_122448/code_0.py
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
#include <c10/cuda/CUDAGuard.h>

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
    float min_value,
    float divisor) {
    
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = batch_size * out_channels * output_depth * output_height * output_width;
    
    if (out_idx >= total_outputs) return;
    
    // Calculate output indices
    int temp = out_idx;
    int w_out = temp % output_width; temp /= output_width;
    int h_out = temp % output_height; temp /= output_height;
    int d_out = temp % output_depth; temp /= output_depth;
    int c_out = temp % out_channels; temp /= out_channels;
    int n = temp;
    
    float result = 0.0f;
    
    // Add bias if provided
    if (bias != nullptr) {
        result = bias[c_out];
    }
    
    // Perform convolution transpose operation
    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int kd = 0; kd < kernel_size; ++kd) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    // Calculate input position
                    int d_in = d_out + padding - kd;
                    int h_in = h_out + padding - kh;
                    int w_in = w_out + padding - kw;
                    
                    // Check bounds and stride alignment
                    if (d_in >= 0 && d_in < input_depth * stride && d_in % stride == 0 &&
                        h_in >= 0 && h_in < input_height * stride && h_in % stride == 0 &&
                        w_in >= 0 && w_in < input_width * stride && w_in % stride == 0) {
                        
                        d_in /= stride;
                        h_in /= stride;
                        w_in /= stride;
                        
                        if (d_in < input_depth && h_in < input_height && w_in < input_width) {
                            int input_idx = n * (in_channels * input_depth * input_height * input_width) +
                                          c_in * (input_depth * input_height * input_width) +
                                          d_in * (input_height * input_width) +
                                          h_in * input_width + w_in;
                            
                            int weight_idx = c_out * (in_channels * kernel_size * kernel_size * kernel_size) +
                                           c_in * (kernel_size * kernel_size * kernel_size) +
                                           kd * (kernel_size * kernel_size) +
                                           kh * kernel_size + kw;
                            
                            result += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
    }
    
    // Apply clamp and division
    result = fmaxf(min_value, result) / divisor;
    
    // Store result
    output[out_idx] = result;
}

void fused_conv_transpose3d_clamp_div(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    at::Tensor& output,
    int stride,
    int padding,
    float min_value,
    float divisor) {
    
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto input_depth = input.size(2);
    auto input_height = input.size(3);
    auto input_width = input.size(4);
    
    auto out_channels = weight.size(0);
    auto kernel_size = weight.size(2);
    
    auto output_depth = (input_depth - 1) * stride - 2 * padding + kernel_size;
    auto output_height = (input_height - 1) * stride - 2 * padding + kernel_size;
    auto output_width = (input_width - 1) * stride - 2 * padding + kernel_size;
    
    int total_outputs = batch_size * out_channels * output_depth * output_height * output_width;
    int threads_per_block = 256;
    int blocks = (total_outputs + threads_per_block - 1) / threads_per_block;
    
    const float* bias_ptr = bias_opt.has_value() ? bias_opt.value().data_ptr<float>() : nullptr;
    
    fused_conv_transpose3d_clamp_div_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_depth,
        input_height,
        input_width,
        output_depth,
        output_height,
        output_width,
        kernel_size,
        stride,
        padding,
        min_value,
        divisor
    );
}
"""

# C++ binding code
cpp_source = r"""
#include <torch/extension.h>

void fused_conv_transpose3d_clamp_div(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    at::Tensor& output,
    int stride,
    int padding,
    float min_value,
    float divisor);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_conv_transpose3d_clamp_div, "Fused conv transpose3d + clamp + div");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_conv_transpose_clamp_div',
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
    # Validate that we can use the fused kernel (simplified constraints)
    assert conv_transpose_groups == 1, "Only groups=1 supported"
    assert conv_transpose_dilation == (1, 1, 1) or conv_transpose_dilation == 1, "Only dilation=1 supported" 
    assert conv_transpose_output_padding == (0, 0, 0) or conv_transpose_output_padding == 0, "Only output_padding=0 supported"
    assert isinstance(conv_transpose_stride, int) or (isinstance(conv_transpose_stride, tuple) and len(conv_transpose_stride) == 3 and all(s == conv_transpose_stride[0] for s in conv_transpose_stride)), "Only uniform stride supported"
    assert isinstance(conv_transpose_padding, int) or (isinstance(conv_transpose_padding, tuple) and len(conv_transpose_padding) == 3 and all(p == conv_transpose_padding[0] for p in conv_transpose_padding)), "Only uniform padding supported"
    
    # Normalize stride and padding to integers
    if isinstance(conv_transpose_stride, tuple):
        stride_val = conv_transpose_stride[0]
    else:
        stride_val = conv_transpose_stride
        
    if isinstance(conv_transpose_padding, tuple):
        padding_val = conv_transpose_padding[0]
    else:
        padding_val = conv_transpose_padding
    
    # Create output tensor with correct shape
    kernel_size = conv_transpose_weight.size(2)
    output_depth = (x.size(2) - 1) * stride_val - 2 * padding_val + kernel_size
    output_height = (x.size(3) - 1) * stride_val - 2 * padding_val + kernel_size  
    output_width = (x.size(4) - 1) * stride_val - 2 * padding_val + kernel_size
    
    output_shape = (x.size(0), conv_transpose_weight.size(0), output_depth, output_height, output_width)
    output = torch.empty(output_shape, device=x.device, dtype=x.dtype)
    
    # Call fused kernel
    fused_ext.fused_op(
        x, 
        conv_transpose_weight, 
        conv_transpose_bias,
        output,
        stride_val,
        padding_val,
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
    return [torch.rand(batch_size, in_channels, depth, height, width, device='cuda')]
