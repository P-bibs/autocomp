# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_125214/code_0.py
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

# Define the fused CUDA kernel
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

__global__ void fused_conv_transpose3d_clamp_divide_kernel(
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
    int output_padding,
    int groups,
    int dilation,
    float min_value,
    float divisor) {
    
    // Calculate global thread indices
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output_elements = batch_size * out_channels * output_depth * output_height * output_width;
    
    if (out_idx >= total_output_elements) return;
    
    // Decompose output index into components
    int temp = out_idx;
    int w_idx = temp % output_width;
    temp /= output_width;
    int h_idx = temp % output_height;
    temp /= output_height;
    int d_idx = temp % output_depth;
    temp /= output_depth;
    int c_idx = temp % out_channels;
    int b_idx = temp / out_channels;
    
    // Calculate input coordinates for transposed convolution
    float sum = 0.0f;
    
    // Group handling
    int group = c_idx / (out_channels / groups);
    int in_channels_per_group = in_channels / groups;
    int out_channels_per_group = out_channels / groups;
    
    // Iterate through kernel
    for (int kd = 0; kd < kernel_size; kd++) {
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                // For transposed convolution, we map output coordinates to input coordinates
                // The relationship is: input_coord = (output_coord + padding - kernel_coord * dilation) / stride
                int in_d = (d_idx + padding - kd * dilation);
                int in_h = (h_idx + padding - kh * dilation);
                int in_w = (w_idx + padding - kw * dilation);
                
                // Check if the input coordinate is valid after division by stride
                if (in_d >= 0 && in_d < input_depth * stride && in_d % stride == 0 &&
                    in_h >= 0 && in_h < input_height * stride && in_h % stride == 0 &&
                    in_w >= 0 && in_w < input_width * stride && in_w % stride == 0) {
                    
                    in_d /= stride;
                    in_h /= stride;
                    in_w /= stride;
                    
                    // Boundary check for input tensor
                    if (in_d >= 0 && in_d < input_depth &&
                        in_h >= 0 && in_h < input_height &&
                        in_w >= 0 && in_w < input_width) {
                        
                        // Iterate through input channels in group
                        for (int ic = 0; ic < in_channels_per_group; ic++) {
                            int input_idx = b_idx * (in_channels * input_depth * input_height * input_width) +
                                          (group * in_channels_per_group + ic) * (input_depth * input_height * input_width) +
                                          in_d * (input_height * input_width) +
                                          in_h * input_width +
                                          in_w;
                                          
                            // Correct weight indexing for ConvTranspose3d
                            // Weight shape: (in_channels, out_channels/groups, kD, kH, kW)
                            int weight_idx = (group * in_channels_per_group + ic) * (out_channels_per_group * kernel_size * kernel_size * kernel_size) +
                                           (c_idx - group * out_channels_per_group) * (kernel_size * kernel_size * kernel_size) +
                                           kd * (kernel_size * kernel_size) +
                                           kh * kernel_size +
                                           kw;
                                           
                            sum += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
    }
    
    // Add bias
    sum += bias[c_idx];
    
    // Apply clamp and division
    sum = fmaxf(sum, min_value);
    sum = sum / divisor;
    
    // Write output
    output[out_idx] = sum;
}

void fused_conv_transpose3d_clamp_divide(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    int groups,
    int dilation,
    float min_value,
    float divisor) {
    
    // Set device guard
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    
    // Get dimensions
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_depth = input.size(2);
    int input_height = input.size(3);
    int input_width = input.size(4);
    
    int out_channels = weight.size(1); // Note: For transposed conv, weight shape is (in_channels, out_channels/groups, kD, kH, kW)
    int output_depth = (input_depth - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1;
    int output_height = (input_height - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1;
    int output_width = (input_width - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1;
    
    // Calculate total output elements
    int total_output_elements = batch_size * out_channels * output_depth * output_height * output_width;
    
    // Configure kernel launch parameters
    const int threads_per_block = 512;
    const int blocks = (total_output_elements + threads_per_block - 1) / threads_per_block;
    
    // Launch kernel
    fused_conv_transpose3d_clamp_divide_kernel<<<blocks, threads_per_block>>>(
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
        output_padding,
        groups,
        dilation,
        min_value,
        divisor
    );
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("CUDA kernel launch failed: ", cudaGetErrorString(err));
    }
}
"""

# Define the C++ interface
cpp_source = r"""
#include <torch/extension.h>

void fused_conv_transpose3d_clamp_divide(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    int groups,
    int dilation,
    float min_value,
    float divisor);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_conv_transpose3d_clamp_divide, "Fused ConvTranspose3d + Clamp + Divide");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_conv_transpose3d_clamp_divide',
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
    batch_size = x.size(0)
    in_channels = x.size(1)
    out_channels = conv_transpose_weight.size(1)
    kernel_size = conv_transpose_weight.size(2)
    
    output_depth = (x.size(2) - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_dilation * (kernel_size - 1) + conv_transpose_output_padding + 1
    output_height = (x.size(3) - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_dilation * (kernel_size - 1) + conv_transpose_output_padding + 1
    output_width = (x.size(4) - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_dilation * (kernel_size - 1) + conv_transpose_output_padding + 1
    
    # Create output tensor
    output = torch.empty((batch_size, out_channels, output_depth, output_height, output_width), 
                         dtype=x.dtype, device=x.device)
    
    # Call fused operation
    fused_ext.fused_op(
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
        min_value,
        divisor
    )
    
    return output

# Test parameters
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
