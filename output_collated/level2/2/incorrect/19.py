# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_163518/code_0.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'output_padding', 'bias_shape', 'scaling_factor']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'bias', 'scaling_factor']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a transposed convolution, adds a bias term, clamps, scales, clamps, and divides.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor

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
    if 'scaling_factor' in flat_state:
        state_kwargs['scaling_factor'] = flat_state['scaling_factor']
    else:
        state_kwargs['scaling_factor'] = getattr(model, 'scaling_factor')
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

# CUDA kernel that fuses all operations into a single kernel
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_op_forward_kernel(
    const float* input,
    const float* conv_transpose_weight,
    const float* conv_transpose_bias,
    const float* bias_tensor,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    int groups,
    int dilation,
    float scaling_factor,
    int out_height,
    int out_width
) {
    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * out_height * out_width;
    
    if (idx >= total_elements) return;
    
    // Calculate indices
    int w_out = idx % out_width;
    int h_out = (idx / out_width) % out_height;
    int c_out = (idx / (out_width * out_width)) % out_channels;
    int b = idx / (out_width * out_height * out_channels);
    
    // Perform transposed convolution computation
    float sum = 0.0f;
    
    // Convolution transpose calculation
    for (int c_in = 0; c_in < in_channels; c_in++) {
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                // Calculate the corresponding input position
                int h_in = (h_out + padding - kh * dilation) / stride;
                int w_in = (w_out + padding - kw * dilation) / stride;
                
                // Check if division was exact and input position is valid
                if ((h_out + padding - kh * dilation) % stride == 0 && 
                    (w_out + padding - kw * dilation) % stride == 0 &&
                    h_in >= 0 && h_in < height && 
                    w_in >= 0 && w_in < width) {
                    
                    int input_idx = ((b * in_channels + c_in) * height + h_in) * width + w_in;
                    int weight_idx = ((c_out * in_channels + c_in) * kernel_size + kh) * kernel_size + kw;
                    
                    sum += input[input_idx] * conv_transpose_weight[weight_idx];
                }
            }
        }
    }
    
    // Add convolution bias if provided
    if (conv_transpose_bias != nullptr) {
        sum += conv_transpose_bias[c_out];
    }
    
    // Add bias tensor
    sum += bias_tensor[c_out];
    
    // First clamp
    sum = fmaxf(0.0f, fminf(1.0f, sum));
    
    // Scale
    sum *= scaling_factor;
    
    // Second clamp
    sum = fmaxf(0.0f, fminf(1.0f, sum));
    
    // Divide by scaling factor
    sum /= scaling_factor;
    
    // Store result
    output[idx] = sum;
}

void fused_op_forward(
    const float* input,
    const float* conv_transpose_weight,
    const float* conv_transpose_bias,
    const float* bias_tensor,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    int groups,
    int dilation,
    float scaling_factor,
    int out_height,
    int out_width,
    int blocks,
    int threads
) {
    fused_op_forward_kernel<<<blocks, threads>>>(
        input,
        conv_transpose_weight,
        conv_transpose_bias,
        bias_tensor,
        output,
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        kernel_size,
        stride,
        padding,
        output_padding,
        groups,
        dilation,
        scaling_factor,
        out_height,
        out_width
    );
}
"""

# C++ binding code
cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(
    const float* input,
    const float* conv_transpose_weight,
    const float* conv_transpose_bias,
    const float* bias_tensor,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    int groups,
    int dilation,
    float scaling_factor,
    int out_height,
    int out_width,
    int blocks,
    int threads
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused operation forward pass");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_op',
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
    scaling_factor,
):
    batch_size, in_channels, height, width = x.shape
    out_channels = conv_transpose_weight.shape[0]
    kernel_size = conv_transpose_weight.shape[2]
    
    # Calculate output dimensions for transposed convolution
    out_height = (height - 1) * conv_transpose_stride + conv_transpose_output_padding - 2 * conv_transpose_padding + kernel_size
    out_width = (width - 1) * conv_transpose_stride + conv_transpose_output_padding - 2 * conv_transpose_padding + kernel_size
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, out_height, out_width, device=x.device, dtype=x.dtype)
    
    # Configure kernel launch parameters
    total_elements = batch_size * out_channels * out_height * out_width
    threads_per_block = 256
    blocks = (total_elements + threads_per_block - 1) // threads_per_block
    
    # Launch fused kernel
    fused_ext.fused_op(
        x.contiguous().data_ptr(),
        conv_transpose_weight.contiguous().data_ptr(),
        conv_transpose_bias.contiguous().data_ptr() if conv_transpose_bias is not None else 0,
        bias.contiguous().data_ptr(),
        output.data_ptr(),
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        kernel_size,
        conv_transpose_stride,
        conv_transpose_padding,
        conv_transpose_output_padding,
        conv_transpose_groups,
        conv_transpose_dilation,
        scaling_factor,
        out_height,
        out_width,
        blocks,
        threads_per_block
    )
    
    return output

# Constants for testing
batch_size = 128
in_channels  = 64  
out_channels = 64  
height = width = 128 
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1)
scaling_factor = 2.0

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]
