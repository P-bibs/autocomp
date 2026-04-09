# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_115905/code_2.py
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

# CUDA kernel for fused conv transpose + bias subtraction + tanh
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__forceinline__ __device__ float tanh_fast(float x) {
    // Fast tanh approximation using rational function
    if (x > 4.0f) return 1.0f;
    if (x < -4.0f) return -1.0f;
    
    float x2 = x * x;
    // Using a rational approximation for tanh
    float numerator = x * (27.0f + x2);
    float denominator = 27.0f + 9.0f * x2;
    return numerator / denominator;
}

__global__ void fused_conv_transpose_bias_tanh_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ subtract_bias,
    float* __restrict__ output,
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
    int out_height,
    int out_width
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * out_height * out_width;
    
    if (tid >= total_elements) return;
    
    int batch_idx = tid / (out_channels * out_height * out_width);
    int remaining = tid % (out_channels * out_height * out_width);
    int out_ch = remaining / (out_height * out_width);
    remaining = remaining % (out_height * out_width);
    int out_h = remaining / out_width;
    int out_w = remaining % out_width;
    
    float sum = 0.0f;
    
    // Calculate corresponding input positions for transpose convolution
    // For each position in the output, we need to check all input positions
    // that could contribute to it through the transpose convolution
    
    // Bounds for input positions that can contribute to this output position
    int h_start = max(0, (out_h - (kernel_size - 1) * dilation + padding) / stride);
    int h_end = min(height, (out_h + padding) / stride + 1);
    int w_start = max(0, (out_w - (kernel_size - 1) * dilation + padding) / stride);
    int w_end = min(width, (out_w + padding) / stride + 1);
    
    // Also make sure we don't exceed bounds due to stride
    h_start = max(h_start, (out_h + padding - (kernel_size - 1) * dilation + stride - 1) / stride);
    w_start = max(w_start, (out_w + padding - (kernel_size - 1) * dilation + stride - 1) / stride);
    
    for (int in_h = h_start; in_h < h_end; in_h++) {
        for (int in_w = w_start; in_w < w_end; in_w++) {
            // Calculate the corresponding kernel position
            int kh = out_h - in_h * stride + padding;
            int kw = out_w - in_w * stride + padding;
            
            // Check if this kernel position is valid
            if (kh >= 0 && kh < kernel_size && kw >= 0 && kw < kernel_size && 
                kh % dilation == 0 && kw % dilation == 0) {
                kh /= dilation;
                kw /= dilation;
                
                // For transpose convolution, we flip the kernel
                int flipped_kh = kernel_size - 1 - kh;
                int flipped_kw = kernel_size - 1 - kw;
                
                for (int in_ch = 0; in_ch < in_channels; ++in_ch) {
                    int input_idx = batch_idx * (in_channels * height * width) + 
                                  in_ch * (height * width) + 
                                  in_h * width + in_w;
                                  
                    int weight_idx = out_ch * (in_channels * kernel_size * kernel_size) + 
                                   in_ch * (kernel_size * kernel_size) + 
                                   flipped_kh * kernel_size + flipped_kw;
                                   
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    // Add convolution bias and subtract the other bias
    sum += conv_bias[out_ch];
    sum -= subtract_bias[out_ch];
    
    // Apply fast tanh
    output[tid] = tanh_fast(sum);
}

void fused_conv_transpose_bias_tanh_forward(
    const torch::Tensor input,
    const torch::Tensor weight,
    const torch::Tensor conv_bias,
    const torch::Tensor subtract_bias,
    torch::Tensor output,
    int stride,
    int padding,
    int output_padding,
    int groups,
    int dilation,
    int kernel_size,
    int out_height,
    int out_width
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    int out_channels = output.size(1);
    
    int total_elements = batch_size * out_channels * out_height * out_width;
    int threads_per_block = 256;
    int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    fused_conv_transpose_bias_tanh_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        conv_bias.data_ptr<float>(),
        subtract_bias.data_ptr<float>(),
        output.data_ptr<float>(),
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
        out_height,
        out_width
    );
}
"""

# C++ binding code
cpp_source = r"""
#include <torch/extension.h>

void fused_conv_transpose_bias_tanh_forward(
    const torch::Tensor input,
    const torch::Tensor weight,
    const torch::Tensor conv_bias,
    const torch::Tensor subtract_bias,
    torch::Tensor output,
    int stride,
    int padding,
    int output_padding,
    int groups,
    int dilation,
    int kernel_size,
    int out_height,
    int out_width
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_conv_transpose_bias_tanh_forward, "Fused Conv Transpose + Bias Subtraction + Tanh");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_conv_tanh',
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
    # Calculate output dimensions
    kernel_size = conv_transpose_weight.shape[2]
    out_height = (x.shape[2] - 1) * conv_transpose_stride[0] - 2 * conv_transpose_padding[0] + kernel_size + conv_transpose_output_padding[0]
    out_width = (x.shape[3] - 1) * conv_transpose_stride[1] - 2 * conv_transpose_padding[1] + kernel_size + conv_transpose_output_padding[1]
    
    # Create output tensor
    output = torch.empty(x.shape[0], conv_transpose_weight.shape[0], out_height, out_width, device=x.device, dtype=x.dtype)
    
    # Call fused CUDA kernel
    fused_ext.fused_op(
        x,
        conv_transpose_weight,
        conv_transpose_bias,
        bias,
        output,
        conv_transpose_stride[0],
        conv_transpose_padding[0],
        conv_transpose_output_padding[0],
        conv_transpose_groups,
        conv_transpose_dilation[0],
        kernel_size,
        out_height,
        out_width
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
