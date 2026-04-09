# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_115905/code_1.py
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
from torch.utils.cpp_extension import load_inline

# CUDA Kernel: Fuses convolution transpose, bias subtraction, and Tanh
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_conv_transpose_tanh_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ activation_bias,
    float* __restrict__ output,
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
    // Calculate output position
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output_elements = batch_size * out_channels * output_height * output_width;
    
    if (tid >= total_output_elements) return;
    
    int n = tid / (out_channels * output_height * output_width);
    int c = (tid / (output_height * output_width)) % out_channels;
    int oh = (tid / output_width) % output_height;
    int ow = tid % output_width;
    
    // Calculate input position that contributes to this output
    float sum = (conv_bias != nullptr) ? conv_bias[c] : 0.0f;
    
    // Perform transposed convolution calculation
    for (int ic = 0; ic < in_channels / groups; ++ic) {
        int group = c / (out_channels / groups);
        if (group * (in_channels / groups) + ic >= in_channels) continue;
        
        int actual_ic = group * (in_channels / groups) + ic;
        
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                // Map output position to input position
                int ih = oh + padding - kh * dilation;
                int iw = ow + padding - kw * dilation;
                
                if (ih % stride == 0 && iw % stride == 0) {
                    ih /= stride;
                    iw /= stride;
                    
                    if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                        int input_idx = n * (in_channels * input_height * input_width) +
                                       actual_ic * (input_height * input_width) +
                                       ih * input_width + iw;
                                       
                        int weight_idx = c * (in_channels / groups * kernel_size * kernel_size) +
                                        ic * (kernel_size * kernel_size) +
                                        kh * kernel_size + kw;
                        
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }
    
    // Apply activation bias subtraction and tanh
    float result = tanhf(sum - activation_bias[c]);
    
    output[tid] = result;
}

void fused_conv_transpose_tanh_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& conv_bias,
    const torch::Tensor& activation_bias,
    torch::Tensor& output,
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
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);
    
    int output_height = (input_height - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1;
    int output_width = (input_width - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1;
    
    int total_output_elements = batch_size * out_channels * output_height * output_width;
    
    // Set up kernel launch parameters
    int threads = 256;
    int blocks = (total_output_elements + threads - 1) / threads;
    
    fused_conv_transpose_tanh_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        conv_bias.defined() ? conv_bias.data_ptr<float>() : nullptr,
        activation_bias.data_ptr<float>(),
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
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_conv_transpose_tanh_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& conv_bias,
    const torch::Tensor& activation_bias,
    torch::Tensor& output,
    int stride,
    int padding,
    int output_padding,
    int groups,
    int dilation
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_transpose_tanh_forward", &fused_conv_transpose_tanh_forward, "Fused ConvTranspose2d with bias subtraction and tanh");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_conv_transpose_tanh',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
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
    batch_size = x.size(0)
    in_channels = x.size(1)
    input_height = x.size(2)
    input_width = x.size(3)
    out_channels = conv_transpose_weight.size(0)
    kernel_size = conv_transpose_weight.size(2)
    
    output_height = (input_height - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_dilation * (kernel_size - 1) + conv_transpose_output_padding + 1
    output_width = (input_width - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_dilation * (kernel_size - 1) + conv_transpose_output_padding + 1
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, output_height, output_width, device=x.device, dtype=x.dtype)
    
    # Flatten bias for kernel usage: bias_shape is (out_channels, 1, 1)
    bias_flat = bias.view(-1)
    
    # Run fused kernel to perform convolution, bias subtraction, and tanh in one pass
    fused_ext.fused_conv_transpose_tanh_forward(
        x, 
        conv_transpose_weight, 
        conv_transpose_bias if conv_transpose_bias is not None else torch.empty(0, device=x.device, dtype=x.dtype),
        bias_flat,
        output,
        conv_transpose_stride,
        conv_transpose_padding,
        conv_transpose_output_padding,
        conv_transpose_groups,
        conv_transpose_dilation
    )
    
    return output

# Setup for testing
batch_size = 32
in_channels = 64
out_channels = 64
height = width = 256
kernel_size = 4
bias_shape = (out_channels, 1, 1)

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]
