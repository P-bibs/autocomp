# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_111953/code_0.py
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
import torch.nn.functional as F

from torch.utils.cpp_extension import load_inline

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_conv_transpose_bias_tanh_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int kernel_size,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int output_padding_h,
    int output_padding_w,
    int groups,
    int dilation_h,
    int dilation_w) {
    
    int batch_idx = blockIdx.x;
    int out_ch = blockIdx.y;
    int thread_idx = blockIdx.z * blockDim.x + threadIdx.x;
    
    if (batch_idx >= batch_size || out_ch >= out_channels) return;
    
    int out_height = (height - 1) * stride_h - 2 * padding_h + (kernel_size - 1) * dilation_h + 1 + output_padding_h;
    int out_width = (width - 1) * stride_w - 2 * padding_w + (kernel_size - 1) * dilation_w + 1 + output_padding_w;
    
    int total_threads = out_height * out_width;
    if (thread_idx >= total_threads) return;
    
    int out_y = thread_idx / out_width;
    int out_x = thread_idx % out_width;
    
    float sum = 0.0f;
    
    // Compute transposed convolution
    for (int in_ch = 0; in_ch < in_channels; ++in_ch) {
        for (int ky = 0; ky < kernel_size; ++ky) {
            for (int kx = 0; kx < kernel_size; ++kx) {
                // Calculate input coordinates
                int in_y = out_y + padding_h - ky * dilation_h;
                int in_x = out_x + padding_w - kx * dilation_w;
                
                // Check if we're in valid input range and aligned with stride
                if (in_y >= 0 && in_y < height * stride_h && in_y % stride_h == 0 &&
                    in_x >= 0 && in_x < width * stride_w && in_x % stride_w == 0) {
                    
                    in_y /= stride_h;
                    in_x /= stride_w;
                    
                    int input_idx = batch_idx * (in_channels * height * width) + 
                                   in_ch * (height * width) + 
                                   in_y * width + in_x;
                                   
                    int weight_idx = out_ch * (in_channels * kernel_size * kernel_size) +
                                    in_ch * (kernel_size * kernel_size) +
                                    ky * kernel_size + kx;
                                    
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    // Add convolution bias
    sum += conv_bias[out_ch];
    
    // Subtract bias and apply tanh
    sum -= bias[out_ch];
    output[batch_idx * (out_channels * out_height * out_width) + 
           out_ch * (out_height * out_width) + 
           out_y * out_width + out_x] = tanhf(sum);
}

void fused_conv_transpose_bias_tanh_forward(
    const torch::Tensor input,
    const torch::Tensor weight,
    const torch::Tensor conv_bias,
    const torch::Tensor bias,
    torch::Tensor output,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int output_padding_h,
    int output_padding_w,
    int groups,
    int dilation_h,
    int dilation_w,
    int kernel_size) {
    
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    int out_channels = weight.size(0);
    
    int out_height = (height - 1) * stride_h - 2 * padding_h + (kernel_size - 1) * dilation_h + 1 + output_padding_h;
    int out_width = (width - 1) * stride_w - 2 * padding_w + (kernel_size - 1) * dilation_w + 1 + output_padding_w;
    
    // Each block handles one output channel of one batch item
    int threads_per_block = 256;
    int total_threads = out_height * out_width;
    int blocks_per_channel = (total_threads + threads_per_block - 1) / threads_per_block;
    
    dim3 grid(batch_size, out_channels, blocks_per_channel);
    dim3 block(threads_per_block);
    
    fused_conv_transpose_bias_tanh_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        conv_bias.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        kernel_size,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        output_padding_h,
        output_padding_w,
        groups,
        dilation_h,
        dilation_w
    );
}
"""

# --- C++ Logic (Interface/Bindings) ---
cpp_source = r"""
#include <torch/extension.h>

// Forward declaration of the function in the .cu file
void fused_conv_transpose_bias_tanh_forward(
    const torch::Tensor input,
    const torch::Tensor weight,
    const torch::Tensor conv_bias,
    const torch::Tensor bias,
    torch::Tensor output,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int output_padding_h,
    int output_padding_w,
    int groups,
    int dilation_h,
    int dilation_w,
    int kernel_size);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_conv_transpose_bias_tanh_forward, "Fused conv transpose, bias subtraction, and tanh");
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
):
    # Create output tensor with correct shape
    batch_size, in_channels, height, width = x.shape
    out_channels = conv_transpose_weight.shape[0]
    kernel_size = conv_transpose_weight.shape[2]
    
    out_height = (height - 1) * conv_transpose_stride[0] - 2 * conv_transpose_padding[0] + (kernel_size - 1) * conv_transpose_dilation[0] + 1 + conv_transpose_output_padding[0]
    out_width = (width - 1) * conv_transpose_stride[1] - 2 * conv_transpose_padding[1] + (kernel_size - 1) * conv_transpose_dilation[1] + 1 + conv_transpose_output_padding[1]
    
    output = torch.empty((batch_size, out_channels, out_height, out_width), device=x.device, dtype=x.dtype)
    
    # Call our fused CUDA kernel
    fused_ext.fused_op(
        x, 
        conv_transpose_weight, 
        conv_transpose_bias, 
        bias.squeeze(), 
        output,
        conv_transpose_stride[0],
        conv_transpose_stride[1],
        conv_transpose_padding[0],
        conv_transpose_padding[1],
        conv_transpose_output_padding[0],
        conv_transpose_output_padding[1],
        conv_transpose_groups,
        conv_transpose_dilation[0],
        conv_transpose_dilation[1],
        kernel_size
    )
    
    return output

batch_size = 32
in_channels  = 64  
out_channels = 64  
height = width = 256 
kernel_size = 4
bias_shape = (out_channels, 1, 1)

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]
