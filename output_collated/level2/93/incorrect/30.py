# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_154319/code_1.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'add_value', 'multiply_value']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'add_value', 'multiply_value']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a transposed convolution, adds a value, takes the minimum, applies GELU, and multiplies by a value.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, add_value, multiply_value):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.add_value = add_value
        self.multiply_value = multiply_value

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
    if 'add_value' in flat_state:
        state_kwargs['add_value'] = flat_state['add_value']
    else:
        state_kwargs['add_value'] = getattr(model, 'add_value')
    if 'multiply_value' in flat_state:
        state_kwargs['multiply_value'] = flat_state['multiply_value']
    else:
        state_kwargs['multiply_value'] = getattr(model, 'multiply_value')
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

# CUDA kernel for fused convolution transpose and post-processing operations
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// CUDA kernel for convolution transpose operation
__global__ void conv_transpose2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int in_height,
    const int in_width,
    const int out_height,
    const int out_width,
    const int kernel_size,
    const int stride,
    const int padding,
    const int output_padding,
    const int groups,
    const int dilation) {
    
    const int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * out_channels * out_height * out_width;
    
    if (out_idx >= total_elements) return;
    
    const int batch = out_idx / (out_channels * out_height * out_width);
    const int oc = (out_idx / (out_height * out_width)) % out_channels;
    const int oh = (out_idx / out_width) % out_height;
    const int ow = out_idx % out_width;
    
    const int group = oc / (out_channels / groups);
    const int weight_offset = group * (in_channels / groups) * out_channels * kernel_size * kernel_size;
    
    float sum = 0.0f;
    
    // Calculate input position
    for (int kh = 0; kh < kernel_size; kh++) {
        for (int kw = 0; kw < kernel_size; kw++) {
            const int ih = oh + padding - kh * dilation;
            const int iw = ow + padding - kw * dilation;
            
            if (ih % stride == 0 && iw % stride == 0) {
                const int src_ih = ih / stride;
                const int src_iw = iw / stride;
                
                if (src_ih >= 0 && src_ih < in_height && src_iw >= 0 && src_iw < in_width) {
                    for (int ic = 0; ic < in_channels / groups; ic++) {
                        const int input_idx = batch * in_channels * in_height * in_width + 
                                            (group * in_channels / groups + ic) * in_height * in_width + 
                                            src_ih * in_width + src_iw;
                        
                        const int weight_idx = weight_offset + 
                                             ic * out_channels * kernel_size * kernel_size + 
                                             oc * kernel_size * kernel_size + 
                                             kh * kernel_size + kw;
                        
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }
    
    output[out_idx] = sum + bias[oc];
}

// CUDA kernel for fused post-processing operations
__global__ void fused_post_conv_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float add_value,
    const float multiply_value,
    const int numel) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < numel) {
        float x = input[idx];
        
        // x = x + add_value
        x += add_value;
        
        // x = torch.min(x, 0.0)
        x = fminf(x, 0.0f);
        
        // x = gelu(x) - using GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        const float sqrt_2_over_pi = 0.7978845608028654f;
        const float coeff = 0.044715f;
        float x_cubed = x * x * x;
        float inner = sqrt_2_over_pi * (x + coeff * x_cubed);
        float tanh_inner = tanhf(inner);
        x = 0.5f * x * (1.0f + tanh_inner);
        
        // x = x * multiply_value
        x *= multiply_value;
        
        output[idx] = x;
    }
}

void fused_conv_transpose_and_post_process_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    const int kernel_size,
    const int stride,
    const int padding,
    const int output_padding,
    const int groups,
    const int dilation,
    const float add_value,
    const float multiply_value) {
    
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int in_height = input.size(2);
    const int in_width = input.size(3);
    const int out_channels = weight.size(1);
    const int out_height = (in_height - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1;
    const int out_width = (in_width - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1;
    
    const int conv_threads_per_block = 256;
    const int conv_total_elements = batch_size * out_channels * out_height * out_width;
    const int conv_blocks = (conv_total_elements + conv_threads_per_block - 1) / conv_threads_per_block;
    
    // Temporary buffer for convolution output
    torch::Tensor temp_output = torch::empty_like(output);
    
    // Launch convolution transpose kernel
    conv_transpose2d_kernel<<<conv_blocks, conv_threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        temp_output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        in_height,
        in_width,
        out_height,
        out_width,
        kernel_size,
        stride,
        padding,
        output_padding,
        groups,
        dilation
    );
    
    // Launch fused post-processing kernel
    const int post_threads_per_block = 256;
    const int post_blocks = (conv_total_elements + post_threads_per_block - 1) / post_threads_per_block;
    
    fused_post_conv_kernel<<<post_blocks, post_threads_per_block>>>(
        temp_output.data_ptr<float>(),
        output.data_ptr<float>(),
        add_value,
        multiply_value,
        conv_total_elements
    );
}
"""

# C++ binding
cpp_source = r"""
#include <torch/extension.h>

void fused_conv_transpose_and_post_process_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    const int kernel_size,
    const int stride,
    const int padding,
    const int output_padding,
    const int groups,
    const int dilation,
    const float add_value,
    const float multiply_value);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_transpose_and_post_process", &fused_conv_transpose_and_post_process_forward, "Fused convolution transpose and post-processing operations");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_conv_transpose_and_post_process',
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
    add_value,
    multiply_value,
):
    # Calculate output dimensions
    batch_size = x.size(0)
    in_channels = x.size(1)
    in_height = x.size(2)
    in_width = x.size(3)
    out_channels = conv_transpose_weight.size(1)
    kernel_size = conv_transpose_weight.size(2)
    
    out_height = (in_height - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_dilation * (kernel_size - 1) + conv_transpose_output_padding + 1
    out_width = (in_width - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_dilation * (kernel_size - 1) + conv_transpose_output_padding + 1
    
    # Create output tensor
    output = torch.empty((batch_size, out_channels, out_height, out_width), device=x.device, dtype=x.dtype)
    
    # Apply fused operations
    fused_ext.fused_conv_transpose_and_post_process(
        x, 
        conv_transpose_weight, 
        conv_transpose_bias, 
        output,
        kernel_size,
        conv_transpose_stride,
        conv_transpose_padding,
        conv_transpose_output_padding,
        conv_transpose_groups,
        conv_transpose_dilation,
        add_value,
        multiply_value
    )
    
    return output

batch_size = 128
in_channels = 64
out_channels = 128
height, width = 64, 64
kernel_size = 4
stride = 2
add_value = 0.5
multiply_value = 2.0

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, add_value, multiply_value]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width, device='cuda')]
