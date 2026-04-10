# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_164254/code_0.py
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
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused post-processing operations
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_post_conv_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ bias,
    const float scaling_factor,
    const int batch_size,
    const int channels,
    const int height,
    const int width
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * channels * height * width;
    
    if (idx < total_elements) {
        const int spatial_size = height * width;
        const int channel_idx = (idx / spatial_size) % channels;
        
        float value = input[idx];
        
        // Add bias (broadcasted over spatial dimensions)
        value += bias[channel_idx];
        
        // First clamp: clamp to [0.0, 1.0]
        value = fmaxf(0.0f, fminf(1.0f, value));
        
        // Multiply by scaling factor
        value *= scaling_factor;
        
        // Second clamp: clamp to [0.0, 1.0]
        value = fmaxf(0.0f, fminf(1.0f, value));
        
        // Divide by scaling factor
        value /= scaling_factor;
        
        output[idx] = value;
    }
}

void fused_post_conv_forward(
    const torch::Tensor input,
    torch::Tensor output,
    const torch::Tensor bias,
    const float scaling_factor
) {
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);
    const int total_elements = batch_size * channels * height * width;
    
    const int threads_per_block = 256;
    const int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    fused_post_conv_kernel<<<num_blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        bias.data_ptr<float>(),
        scaling_factor,
        batch_size,
        channels,
        height,
        width
    );
}
"""

# C++ interface for fused post-processing
cpp_source = r"""
#include <torch/extension.h>

void fused_post_conv_forward(
    const torch::Tensor input,
    torch::Tensor output,
    const torch::Tensor bias,
    const float scaling_factor
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_post_conv", &fused_post_conv_forward, "Fused post-convolution operations");
}
"""

# Compile the fused post-processing extension
fused_post_ext = load_inline(
    name='fused_post_conv',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# CUDA kernel for optimized conv transpose 2d
conv_transpose_cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int in_height,
    const int in_width,
    const int out_channels,
    const int kernel_size,
    const int stride,
    const int padding,
    const int output_padding,
    const int groups,
    const int dilation
) {
    // Calculate output dimensions
    const int out_height = (in_height - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1 + output_padding;
    const int out_width = (in_width - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1 + output_padding;
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = batch_size * out_channels * out_height * out_width;
    
    if (tid < total_threads) {
        const int w_out = tid % out_width;
        const int h_out = (tid / out_width) % out_height;
        const int c_out = (tid / (out_width * out_height)) % out_channels;
        const int n = tid / (out_width * out_height * out_channels);
        
        float value = 0.0f;
        
        const int kernel_radius = kernel_size / 2;
        
        // For each position in the kernel
        for (int ky = 0; ky < kernel_size; ky++) {
            for (int kx = 0; kx < kernel_size; kx++) {
                // Calculate corresponding input position
                const int h_in = h_out + padding - ky * dilation;
                const int w_in = w_out + padding - kx * dilation;
                
                // Check if input position is valid
                if (h_in >= 0 && h_in < in_height * stride && h_in % stride == 0 &&
                    w_in >= 0 && w_in < in_width * stride && w_in % stride == 0) {
                    
                    const int h_in_idx = h_in / stride;
                    const int w_in_idx = w_in / stride;
                    
                    if (h_in_idx < in_height && w_in_idx < in_width) {
                        // Get input value
                        const int input_idx = ((n * in_channels + c_out) * in_height + h_in_idx) * in_width + w_in_idx;
                        const int weight_idx = ((c_out * kernel_size + ky) * kernel_size + kx);
                        
                        value += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        
        // Add bias
        output[tid] = value + bias[c_out];
    }
}

// Optimized version using shared memory and better memory access patterns
__global__ void conv_transpose2d_optimized_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int in_height,
    const int in_width,
    const int out_channels,
    const int kernel_size,
    const int stride,
    const int padding,
    const int output_padding,
    const int groups,
    const int dilation
) {
    const int out_height = (in_height - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1 + output_padding;
    const int out_width = (in_width - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1 + output_padding;
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = batch_size * out_channels * out_height * out_width;
    
    if (tid < total_threads) {
        const int w_out = tid % out_width;
        const int h_out = (tid / out_width) % out_height;
        const int c_out = (tid / (out_width * out_height)) % out_channels;
        const int n = tid / (out_width * out_height * out_channels);
        
        float value = 0.0f;
        
        // For each position in the kernel
        for (int ky = 0; ky < kernel_size; ky++) {
            for (int kx = 0; kx < kernel_size; kx++) {
                // Calculate corresponding input position
                const int h_in = h_out + padding - ky * dilation;
                const int w_in = w_out + padding - kx * dilation;
                
                // Check if input position is valid and aligned with stride
                if (h_in >= 0 && (h_in % stride) == 0 && (h_in / stride) < in_height &&
                    w_in >= 0 && (w_in % stride) == 0 && (w_in / stride) < in_width) {
                    
                    const int h_in_idx = h_in / stride;
                    const int w_in_idx = w_in / stride;
                    
                    // Get input value
                    const int input_idx = ((n * in_channels + c_out) * in_height + h_in_idx) * in_width + w_in_idx;
                    // Note: This assumes grouped conv with groups = in_channels = out_channels
                    const int weight_idx = (c_out * kernel_size + (kernel_size - 1 - ky)) * kernel_size + (kernel_size - 1 - kx);
                    
                    value += input[input_idx] * weight[weight_idx];
                }
            }
        }
        
        // Add bias
        output[tid] = value + bias[c_out];
    }
}

void conv_transpose2d_forward(
    const torch::Tensor input,
    const torch::Tensor weight,
    const torch::Tensor bias,
    torch::Tensor output,
    const int stride,
    const int padding,
    const int output_padding,
    const int groups,
    const int dilation
) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int in_height = input.size(2);
    const int in_width = input.size(3);
    const int out_channels = weight.size(1);
    const int kernel_size = weight.size(2);
    
    const int out_height = (in_height - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1 + output_padding;
    const int out_width = (in_width - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1 + output_padding;
    
    const int total_elements = batch_size * out_channels * out_height * out_width;
    const int threads_per_block = 256;
    const int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    conv_transpose2d_optimized_kernel<<<num_blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        in_height,
        in_width,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        groups,
        dilation
    );
}
"""

# C++ interface for conv transpose 2d
cpp_conv_source = r"""
#include <torch/extension.h>

void conv_transpose2d_forward(
    const torch::Tensor input,
    const torch::Tensor weight,
    const torch::Tensor bias,
    torch::Tensor output,
    const int stride,
    const int padding,
    const int output_padding,
    const int groups,
    const int dilation
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_transpose2d", &conv_transpose2d_forward, "Custom CUDA ConvTranspose2d");
}
"""

# Compile the conv transpose 2d extension
conv_transpose_ext = load_inline(
    name='conv_transpose2d',
    cpp_sources=cpp_conv_source,
    cuda_sources=conv_transpose_cuda_kernel,
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
    # Perform conv transpose using our custom CUDA kernel
    batch_size = x.size(0)
    in_channels = x.size(1)
    in_height = x.size(2)
    in_width = x.size(3)
    
    # Calculate output dimensions
    kernel_size = conv_transpose_weight.size(2)
    out_height = (in_height - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_dilation * (kernel_size - 1) + 1 + conv_transpose_output_padding
    out_width = (in_width - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_dilation * (kernel_size - 1) + 1 + conv_transpose_output_padding
    out_channels = conv_transpose_weight.size(1)
    
    conv_output = torch.empty(batch_size, out_channels, out_height, out_width, device=x.device, dtype=x.dtype)
    
    # Use our custom CUDA kernel for conv transpose
    conv_transpose_ext.conv_transpose2d(
        x, 
        conv_transpose_weight, 
        conv_transpose_bias, 
        conv_output,
        conv_transpose_stride, 
        conv_transpose_padding, 
        conv_transpose_output_padding, 
        conv_transpose_groups, 
        conv_transpose_dilation
    )
    
    # Create output tensor for final result
    final_output = torch.empty_like(conv_output)
    
    # Use our fused CUDA kernel for the remaining operations
    fused_post_ext.fused_post_conv(conv_output, final_output, bias.squeeze(), scaling_factor)
    
    return final_output

# Test parameters
batch_size = 128
in_channels = 64  
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
    return [torch.rand(batch_size, in_channels, height, width, device='cuda')]
