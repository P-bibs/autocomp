# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_115141/code_2.py
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

# Custom CUDA kernel for fused conv transpose 3d + clamp + div
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

__global__ void fused_conv_transpose3d_clamp_div_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
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
    float min_value,
    float divisor) {
    
    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * output_depth * output_height * output_width;
    
    if (idx >= total_elements) return;
    
    // Decode output position
    int temp = idx;
    int out_w = temp % output_width;
    temp /= output_width;
    int out_h = temp % output_height;
    temp /= output_height;
    int out_d = temp % output_depth;
    temp /= output_depth;
    int out_c = temp % out_channels;
    int batch = temp / out_channels;
    
    // Calculate convolution sum
    float sum = 0.0f;
    
    // Determine kernel bounds
    int k_d_min = fmaxf(0, (out_d + padding - kernel_size + 1 + stride - 1) / stride);
    int k_d_max = fminf(kernel_size - 1, (out_d + padding) / stride);
    int k_h_min = fmaxf(0, (out_h + padding - kernel_size + 1 + stride - 1) / stride);
    int k_h_max = fminf(kernel_size - 1, (out_h + padding) / stride);
    int k_w_min = fmaxf(0, (out_w + padding - kernel_size + 1 + stride - 1) / stride);
    int k_w_max = fminf(kernel_size - 1, (out_w + padding) / stride);
    
    for (int k_d = k_d_min; k_d <= k_d_max; k_d++) {
        for (int k_h = k_h_min; k_h <= k_h_max; k_h++) {
            for (int k_w = k_w_min; k_w <= k_w_max; k_w++) {
                int in_d = out_d + padding - k_d * stride;
                int in_h = out_h + padding - k_h * stride;
                int in_w = out_w + padding - k_w * stride;
                
                if (in_d >= 0 && in_d < input_depth &&
                    in_h >= 0 && in_h < input_height &&
                    in_w >= 0 && in_w < input_width) {
                    
                    for (int in_c = 0; in_c < in_channels; in_c++) {
                        int input_idx = batch * (in_channels * input_depth * input_height * input_width) +
                                       in_c * (input_depth * input_height * input_width) +
                                       in_d * (input_height * input_width) +
                                       in_h * input_width +
                                       in_w;
                                       
                        int weight_idx = out_c * (in_channels * kernel_size * kernel_size * kernel_size) +
                                        in_c * (kernel_size * kernel_size * kernel_size) +
                                        k_d * (kernel_size * kernel_size) +
                                        k_h * kernel_size +
                                        k_w;
                                        
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }
    
    // Add bias
    sum += bias[out_c];
    
    // Apply clamp and division
    sum = fmaxf(sum, min_value);
    sum = sum / divisor;
    
    // Write result
    output[idx] = sum;
}

void fused_conv_transpose3d_clamp_div_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int kernel_size,
    int stride,
    int padding,
    float min_value,
    float divisor) {
    
    // Get tensor dimensions
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto input_depth = input.size(2);
    auto input_height = input.size(3);
    auto input_width = input.size(4);
    
    auto out_channels = weight.size(0);
    auto output_depth = output.size(2);
    auto output_height = output.size(3);
    auto output_width = output.size(4);
    
    // Calculate total number of output elements
    int total_elements = batch_size * out_channels * output_depth * output_height * output_width;
    
    // Configure kernel launch parameters
    const int threads_per_block = 512;
    const int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    // Set CUDA device guard
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    
    // Launch kernel
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
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("CUDA kernel launch failed: ", cudaGetErrorString(err));
    }
}
"""

# C++ binding code
cpp_source = r"""
#include <torch/extension.h>

void fused_conv_transpose3d_clamp_div_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int kernel_size,
    int stride,
    int padding,
    float min_value,
    float divisor);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_transpose3d_clamp_div_forward", 
          &fused_conv_transpose3d_clamp_div_forward, 
          "Fused ConvTranspose3d + Clamp + Div forward pass");
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
    # Calculate output dimensions for conv transpose
    batch_size, in_channels, depth, height, width = x.shape
    out_channels = conv_transpose_weight.shape[0]
    kernel_size = conv_transpose_weight.shape[2]
    
    # Compute output dimensions
    output_depth = (depth - 1) * conv_transpose_stride + conv_transpose_dilation * (kernel_size - 1) + 1 - 2 * conv_transpose_padding + conv_transpose_output_padding
    output_height = (height - 1) * conv_transpose_stride + conv_transpose_dilation * (kernel_size - 1) + 1 - 2 * conv_transpose_padding + conv_transpose_output_padding
    output_width = (width - 1) * conv_transpose_stride + conv_transpose_dilation * (kernel_size - 1) + 1 - 2 * conv_transpose_padding + conv_transpose_output_padding
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, output_depth, output_height, output_width, 
                         dtype=x.dtype, device=x.device)
    
    # Call fused kernel
    fused_ext.fused_conv_transpose3d_clamp_div_forward(
        x, conv_transpose_weight, conv_transpose_bias, output,
        kernel_size, conv_transpose_stride, conv_transpose_padding,
        min_value, divisor
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
    return [torch.rand(batch_size, in_channels, depth, height, width)]
