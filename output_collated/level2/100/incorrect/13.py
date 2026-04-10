# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_120759/code_2.py
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
import math

# Custom CUDA kernel for fused conv_transpose3d + clamp + division
from torch.utils.cpp_extension import load_inline

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
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    int groups,
    int dilation,
    float min_value,
    float divisor,
    int output_depth,
    int output_height,
    int output_width
) {
    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * output_depth * output_height * output_width;
    
    if (idx >= total_elements) return;
    
    // Decompose linear index into multidimensional indices
    int temp = idx;
    int w = temp % output_width;
    temp /= output_width;
    int h = temp % output_height;
    temp /= output_height;
    int d = temp % output_depth;
    temp /= output_depth;
    int c = temp % out_channels;
    int b = temp / out_channels;
    
    // Calculate convolution transpose value
    float sum = 0.0f;
    
    // Determine kernel dimensions accounting for dilation
    int dilated_kernel_size = dilation * (kernel_size - 1) + 1;
    
    // Determine input range that affects this output position
    int start_kd = max(0, (d + padding - dilated_kernel_size + 1 + stride - 1) / stride);
    int end_kd = min(input_depth, (d + padding + stride) / stride);
    int start_kh = max(0, (h + padding - dilated_kernel_size + 1 + stride - 1) / stride);
    int end_kh = min(input_height, (h + padding + stride) / stride);
    int start_kw = max(0, (w + padding - dilated_kernel_size + 1 + stride - 1) / stride);
    int end_kw = min(input_width, (w + padding + stride) / stride);
    
    // Iterate through input points and kernel points
    for (int kd = start_kd; kd < end_kd; ++kd) {
        for (int kh = start_kh; kh < end_kh; ++kh) {
            for (int kw = start_kw; kw < end_kw; ++kw) {
                // Calculate kernel indices
                int kd_k = (d + padding - stride * kd) / dilation;
                int kh_k = (h + padding - stride * kh) / dilation;
                int kw_k = (w + padding - stride * kw) / dilation;
                
                // Check if kernel indices are valid
                if (kd_k >= 0 && kd_k < kernel_size &&
                    kh_k >= 0 && kh_k < kernel_size &&
                    kw_k >= 0 && kw_k < kernel_size) {
                    
                    // Calculate channel group
                    int group = c / (out_channels / groups);
                    int in_ch_start = group * (in_channels / groups);
                    int in_ch_end = (group + 1) * (in_channels / groups);
                    
                    // Accumulate over input channels in the group
                    for (int ic = in_ch_start; ic < in_ch_end; ++ic) {
                        // Input index
                        int input_idx = b * (in_channels * input_depth * input_height * input_width) +
                                        ic * (input_depth * input_height * input_width) +
                                        kd * (input_height * input_width) +
                                        kh * input_width +
                                        kw;
                        
                        // Weight index
                        int weight_idx = c * (in_channels / groups * kernel_size * kernel_size * kernel_size) +
                                         (ic - in_ch_start) * (kernel_size * kernel_size * kernel_size) +
                                         kd_k * (kernel_size * kernel_size) +
                                         kh_k * kernel_size +
                                         kw_k;
                        
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }
    
    // Add bias if provided
    if (bias != nullptr) {
        sum += bias[c];
    }
    
    // Apply clamp and division
    float result = fmaxf(sum, min_value) / divisor;
    
    // Write output
    output[idx] = result;
}

void fused_conv_transpose3d_clamp_div_forward(
    const torch::Tensor input,
    const torch::Tensor weight,
    const torch::Tensor bias,
    torch::Tensor output,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    int groups,
    int dilation,
    float min_value,
    float divisor
) {
    // Set CUDA device guard
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    
    // Get tensor dimensions
    auto input_sizes = input.sizes();
    auto weight_sizes = weight.sizes();
    auto output_sizes = output.sizes();
    
    int batch_size = input_sizes[0];
    int in_channels = input_sizes[1];
    int input_depth = input_sizes[2];
    int input_height = input_sizes[3];
    int input_width = input_sizes[4];
    
    int out_channels = weight_sizes[0];
    int output_depth = output_sizes[2];
    int output_height = output_sizes[3];
    int output_width = output_sizes[4];
    
    // Calculate grid and block dimensions
    int total_elements = batch_size * out_channels * output_depth * output_height * output_width;
    int threads_per_block = 256;
    int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    // Launch kernel
    fused_conv_transpose3d_clamp_div_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_depth,
        input_height,
        input_width,
        kernel_size,
        stride,
        padding,
        output_padding,
        groups,
        dilation,
        min_value,
        divisor,
        output_depth,
        output_height,
        output_width
    );
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("CUDA kernel launch failed: ", cudaGetErrorString(err));
    }
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
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    int groups,
    int dilation,
    float min_value,
    float divisor
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_transpose3d_clamp_div_forward", 
          &fused_conv_transpose3d_clamp_div_forward, 
          "Fused ConvTranspose3d + Clamp + Division forward");
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
    
    # Calculate output dimensions for conv_transpose3d
    output_depth = (input_depth - 1) * conv_transpose_stride + conv_transpose_dilation * (kernel_size - 1) + 1 - 2 * conv_transpose_padding + conv_transpose_output_padding
    output_height = (input_height - 1) * conv_transpose_stride + conv_transpose_dilation * (kernel_size - 1) + 1 - 2 * conv_transpose_padding + conv_transpose_output_padding
    output_width = (input_width - 1) * conv_transpose_stride + conv_transpose_dilation * (kernel_size - 1) + 1 - 2 * conv_transpose_padding + conv_transpose_output_padding
    
    # Create output tensor
    output = torch.empty((batch_size, out_channels, output_depth, output_height, output_width), 
                         dtype=x.dtype, device=x.device)
    
    # Call fused kernel
    fused_ext.fused_conv_transpose3d_clamp_div_forward(
        x.contiguous(),
        conv_transpose_weight.contiguous(),
        conv_transpose_bias.contiguous() if conv_transpose_bias is not None else torch.tensor([]),
        output,
        kernel_size,
        conv_transpose_stride,
        conv_transpose_padding,
        conv_transpose_output_padding,
        conv_transpose_groups,
        conv_transpose_dilation,
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
