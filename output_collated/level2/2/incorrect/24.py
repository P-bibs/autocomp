# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_163844/code_0.py
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

# CUDA kernel code implementing fused conv_transpose2d + bias + clamp + scale operations
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

__global__ void fused_conv_transpose_bias_clamp_scale_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int in_height,
    const int in_width,
    const int out_height,
    const int out_width,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int padding_h,
    const int padding_w,
    const int output_padding_h,
    const int output_padding_w,
    const float scaling_factor
) {
    // Calculate output indices
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * out_height * out_width;
    
    if (idx >= total_elements) return;
    
    // Decode output tensor indices
    int temp = idx;
    int w_out = temp % out_width;
    temp /= out_width;
    int h_out = temp % out_height;
    temp /= out_height;
    int c_out = temp % out_channels;
    int n = temp / out_channels;
    
    float sum = 0.0f;
    
    // Perform transposed convolution
    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                // Calculate the corresponding input position
                int h_in = h_out - kh + kernel_h - 1 - padding_h;
                int w_in = w_out - kw + kernel_w - 1 - padding_w;
                
                // Check if the input position is valid after considering stride
                if ((h_in % stride_h == 0) && (w_in % stride_w == 0)) {
                    h_in /= stride_h;
                    w_in /= stride_w;
                    
                    // Check bounds
                    if (h_in >= 0 && h_in < in_height && w_in >= 0 && w_in < in_width) {
                        int input_idx = ((n * in_channels + c_in) * in_height + h_in) * in_width + w_in;
                        int weight_idx = ((c_in * out_channels + c_out) * kernel_h + (kernel_h - 1 - kh)) * kernel_w + (kernel_w - 1 - kw);
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }
    
    // Add convolution bias
    sum += conv_bias[c_out];
    
    // Apply first bias
    sum += bias[c_out];
    
    // Apply first clamp
    sum = fmaxf(0.0f, fminf(1.0f, sum));
    
    // Scale
    sum *= scaling_factor;
    
    // Apply second clamp
    sum = fmaxf(0.0f, fminf(1.0f, sum));
    
    // Unscale
    sum /= scaling_factor;
    
    output[idx] = sum;
}

void fused_conv_transpose_bias_clamp_scale_op(
    const torch::Tensor input,
    const torch::Tensor weight,
    const torch::Tensor conv_bias,
    const torch::Tensor bias,
    torch::Tensor output,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int padding_h,
    const int padding_w,
    const int output_padding_h,
    const int output_padding_w,
    const float scaling_factor
) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_height = input.size(2);
    int in_width = input.size(3);
    int out_channels = output.size(1);
    int out_height = output.size(2);
    int out_width = output.size(3);
    
    int total_elements = batch_size * out_channels * out_height * out_width;
    int threads_per_block = 256;
    int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    fused_conv_transpose_bias_clamp_scale_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        conv_bias.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        in_height,
        in_width,
        out_height,
        out_width,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        output_padding_h,
        output_padding_w,
        scaling_factor
    );
    
    cudaDeviceSynchronize();
}
"""

# C++ binding code
cpp_source = r"""
#include <torch/extension.h>

void fused_conv_transpose_bias_clamp_scale_op(
    const torch::Tensor input,
    const torch::Tensor weight,
    const torch::Tensor conv_bias,
    const torch::Tensor bias,
    torch::Tensor output,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int padding_h,
    const int padding_w,
    const int output_padding_h,
    const int output_padding_w,
    const float scaling_factor
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_conv_transpose_bias_clamp_scale_op, "Fused conv transpose bias clamp scale operation");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_conv_transpose_op',
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
    # Calculate output dimensions for conv_transpose2d
    batch_size, in_channels, in_height, in_width = x.shape
    out_channels = conv_transpose_weight.shape[1]  # Note: for conv_transpose, weight shape is (in_channels, out_channels/groups, kH, kW)
    
    # Calculate output spatial dimensions
    out_height = (in_height - 1) * conv_transpose_stride[0] - 2 * conv_transpose_padding[0] + conv_transpose_weight.shape[2] + conv_transpose_output_padding[0]
    out_width = (in_width - 1) * conv_transpose_stride[1] - 2 * conv_transpose_padding[1] + conv_transpose_weight.shape[3] + conv_transpose_output_padding[1]
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, out_height, out_width, device=x.device, dtype=x.dtype)
    
    # Call fused CUDA kernel
    fused_ext.fused_op(
        x.contiguous(),
        conv_transpose_weight.contiguous(),
        conv_transpose_bias.contiguous(),
        bias.contiguous(),
        output,
        conv_transpose_weight.shape[2],  # kernel_h
        conv_transpose_weight.shape[3],  # kernel_w
        conv_transpose_stride[0],
        conv_transpose_stride[1],
        conv_transpose_padding[0],
        conv_transpose_padding[1],
        conv_transpose_output_padding[0],
        conv_transpose_output_padding[1],
        float(scaling_factor)
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
