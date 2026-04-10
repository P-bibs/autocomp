# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_130522/code_2.py
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

# Custom CUDA kernel for fused clamp and division operation
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_conv_transpose3d_clamp_div_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int input_depth,
    const int input_height,
    const int input_width,
    const int output_depth,
    const int output_height,
    const int output_width,
    const int kernel_size,
    const int stride,
    const int padding,
    const float min_value,
    const float divisor
) {
    // Grid-stride loop
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * output_depth * output_height * output_width;
    
    // Shared memory for weights (if fits)
    extern __shared__ float shared_weight[];
    
    while (idx < total_elements) {
        int w = idx % output_width;
        idx /= output_width;
        int h = idx % output_height;
        idx /= output_height;
        int d = idx % output_depth;
        idx /= output_depth;
        int oc = idx % out_channels;
        idx /= out_channels;
        int b = idx;
        
        float sum = 0.0f;
        
        // Calculate convolution transpose
        for (int ic = 0; ic < in_channels; ++ic) {
            for (int kd = 0; kd < kernel_size; ++kd) {
                for (int kh = 0; kh < kernel_size; ++kh) {
                    for (int kw = 0; kw < kernel_size; ++kw) {
                        // Map output position to input position
                        int in_d = d - kd + padding;
                        int in_h = h - kh + padding;
                        int in_w = w - kw + padding;
                        
                        // Check if in valid input range
                        if (in_d >= 0 && in_d < input_depth*stride && in_d % stride == 0 &&
                            in_h >= 0 && in_h < input_height*stride && in_h % stride == 0 &&
                            in_w >= 0 && in_w < input_width*stride && in_w % stride == 0) {
                            
                            int orig_d = in_d / stride;
                            int orig_h = in_h / stride;
                            int orig_w = in_w / stride;
                            
                            int input_idx = ((((b * in_channels) + ic) * input_depth + orig_d) * input_height + orig_h) * input_width + orig_w;
                            int weight_idx = ((((oc * in_channels) + ic) * kernel_size + kd) * kernel_size + kh) * kernel_size + kw;
                            
                            sum += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
        
        // Add bias
        sum += bias[oc];
        
        // Apply clamp and division
        sum = fmaxf(sum, min_value);
        sum = sum / divisor;
        
        output[idx] = sum;
        
        // Grid-stride loop increment
        idx += blockDim.x * gridDim.x;
    }
}

void fused_conv_transpose3d_clamp_div_forward(
    const torch::Tensor input,
    const torch::Tensor weight,
    const torch::Tensor bias,
    torch::Tensor output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int input_depth,
    const int input_height,
    const int input_width,
    const int output_depth,
    const int output_height,
    const int output_width,
    const int kernel_size,
    const int stride,
    const int padding,
    const float min_value,
    const float divisor
) {
    const int threads_per_block = 256;
    const int blocks = (batch_size * out_channels * output_depth * output_height * output_width + threads_per_block - 1) / threads_per_block;
    
    fused_conv_transpose3d_clamp_div_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
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
    
    cudaDeviceSynchronize();
}
"""

# C++ interface
cpp_source = r"""
#include <torch/extension.h>

void fused_conv_transpose3d_clamp_div_forward(
    const torch::Tensor input,
    const torch::Tensor weight,
    const torch::Tensor bias,
    torch::Tensor output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int input_depth,
    const int input_height,
    const int input_width,
    const int output_depth,
    const int output_height,
    const int output_width,
    const int kernel_size,
    const int stride,
    const int padding,
    const float min_value,
    const float divisor
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_transpose3d_clamp_div", &fused_conv_transpose3d_clamp_div_forward, "Fused 3D Conv Transpose with Clamp and Division");
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
    # Calculate output dimensions
    batch_size, in_channels, input_depth, input_height, input_width = x.shape
    out_channels = conv_transpose_weight.shape[0]
    kernel_size = conv_transpose_weight.shape[2]
    
    # Compute output dimensions for transposed convolution
    output_depth = (input_depth - 1) * conv_transpose_stride + conv_transpose_dilation * (kernel_size - 1) + 1 - 2 * conv_transpose_padding + conv_transpose_output_padding
    output_height = (input_height - 1) * conv_transpose_stride + conv_transpose_dilation * (kernel_size - 1) + 1 - 2 * conv_transpose_padding + conv_transpose_output_padding
    output_width = (input_width - 1) * conv_transpose_stride + conv_transpose_dilation * (kernel_size - 1) + 1 - 2 * conv_transpose_padding + conv_transpose_output_padding
    
    # Create output tensor
    output = torch.empty((batch_size, out_channels, output_depth, output_height, output_width), device=x.device, dtype=x.dtype)
    
    # Call fused kernel
    fused_ext.fused_conv_transpose3d_clamp_div(
        x, conv_transpose_weight, conv_transpose_bias, output,
        batch_size, in_channels, out_channels,
        input_depth, input_height, input_width,
        output_depth, output_height, output_width,
        kernel_size, conv_transpose_stride, conv_transpose_padding,
        min_value, divisor
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
