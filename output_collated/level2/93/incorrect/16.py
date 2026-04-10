# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_153448/code_1.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'add_value', 'multiply_value']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'add_value', 'multiply_value']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a transposed convolution, adds a value, takes the minimum, applies GELU, and multiplies by a value.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, add_value, multiply_value):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.add_value = add_value
        self.multiply_value = multiply_value

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
    if 'add_value' in flat_state:
        state_kwargs['add_value'] = flat_state['add_value']
    else:
        state_kwargs['add_value'] = getattr(model, 'add_value')
    if 'multiply_value' in flat_state:
        state_kwargs['multiply_value'] = flat_state['multiply_value']
    else:
        state_kwargs['multiply_value'] = getattr(model, 'multiply_value')
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

# Define CUDA kernel source
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

__global__ void fused_conv_transpose_activation_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float add_value,
    const float multiply_value,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int in_height,
    const int in_width,
    const int out_height,
    const int out_width,
    const int kernel_size,
    const int stride,
    const int padding,
    const int output_padding,
    const int groups,
    const int dilation) {
    
    // Calculate output indices
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output_elements = batch_size * out_channels * out_height * out_width;
    
    if (out_idx >= total_output_elements) return;
    
    // Decode output indices
    int tmp = out_idx;
    int w_out = tmp % out_width; tmp /= out_width;
    int h_out = tmp % out_height; tmp /= out_height;
    int c_out = tmp % out_channels; tmp /= out_channels;
    int n = tmp;
    
    // Calculate input position
    float accumulator = 0.0f;
    
    // Conv transpose calculation
    int kernel_h = kernel_size;
    int kernel_w = kernel_size;
    
    // Iterate through kernel
    for (int kh = 0; kh < kernel_h; kh++) {
        for (int kw = 0; kw < kernel_w; kw++) {
            // Calculate corresponding input position
            int h_in = (h_out + padding - kh * dilation) / stride;
            int w_in = (w_out + padding - kw * dilation) / stride;
            
            // Check if division was exact (valid position)
            if ((h_out + padding - kh * dilation) % stride == 0 && 
                (w_out + padding - kw * dilation) % stride == 0 &&
                h_in >= 0 && h_in < in_height && 
                w_in >= 0 && w_in < in_width) {
                
                // Calculate weight index (assuming weight is [in_channels, out_channels/groups, kernel_h, kernel_w])
                int group = c_out / (out_channels / groups);
                int c_in_start = group * (in_channels / groups);
                int c_in_end = (group + 1) * (in_channels / groups);
                
                for (int c_in = c_in_start; c_in < c_in_end; c_in++) {
                    int weight_idx = ((c_in * (out_channels/groups) + (c_out % (out_channels/groups))) * kernel_h + kh) * kernel_w + kw;
                    int input_idx = ((n * in_channels + c_in) * in_height + h_in) * in_width + w_in;
                    accumulator += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    // Add bias
    accumulator += bias[c_out];
    
    // Add value
    accumulator += add_value;
    
    // Min with 0
    accumulator = fminf(accumulator, 0.0f);
    
    // GELU activation: GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
    float gelu_input = accumulator;
    accumulator = 0.5f * gelu_input * (1.0f + erff(gelu_input * 0.70710678118654752440f)); // 1/sqrt(2) ≈ 0.70710678118654752440
    
    // Multiply by value
    accumulator *= multiply_value;
    
    // Store result
    output[out_idx] = accumulator;
}

void fused_conv_transpose_activation_forward(
    const torch::Tensor input,
    const torch::Tensor weight,
    const torch::Tensor bias,
    const float add_value,
    const float multiply_value,
    torch::Tensor output,
    const int kernel_size,
    const int stride,
    const int padding,
    const int output_padding,
    const int groups,
    const int dilation) {
    
    // Set CUDA device guard
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    
    // Get dimensions
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_height = input.size(2);
    int in_width = input.size(3);
    
    int out_channels = output.size(1);
    int out_height = output.size(2);
    int out_width = output.size(3);
    
    // Launch configuration
    int total_output_elements = batch_size * out_channels * out_height * out_width;
    int threads_per_block = 256;
    int blocks = (total_output_elements + threads_per_block - 1) / threads_per_block;
    
    // Launch kernel
    fused_conv_transpose_activation_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        add_value,
        multiply_value,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        in_height,
        in_width,
        out_height,
        out_width,
        kernel_size,
        stride,
        padding,
        output_padding,
        groups,
        dilation
    );
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("CUDA kernel launch failed: ", cudaGetErrorString(err));
    }
}
"""

# Define C++ interface
cpp_source = r"""
#include <torch/extension.h>

void fused_conv_transpose_activation_forward(
    const torch::Tensor input,
    const torch::Tensor weight,
    const torch::Tensor bias,
    const float add_value,
    const float multiply_value,
    torch::Tensor output,
    const int kernel_size,
    const int stride,
    const int padding,
    const int output_padding,
    const int groups,
    const int dilation);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_transpose_activation", 
          &fused_conv_transpose_activation_forward, 
          "Fused ConvTranspose2d + Add + Min + GELU + Multiply");
}
"""

# Compile extension
fused_ext = load_inline(
    name='fused_conv_transpose_activation_ext',
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
    add_value,
    multiply_value,
):
    # Calculate output dimensions
    batch_size, in_channels, in_height, in_width = x.shape
    out_channels = conv_transpose_weight.size(1) * conv_transpose_groups  # Adjusted for grouped conv
    kernel_size = conv_transpose_weight.size(2)
    
    # Calculate output spatial dimensions for conv transpose
    out_height = (in_height - 1) * conv_transpose_stride + (kernel_size - 1) * conv_transpose_dilation + 1 - 2 * conv_transpose_padding + conv_transpose_output_padding
    out_width = (in_width - 1) * conv_transpose_stride + (kernel_size - 1) * conv_transpose_dilation + 1 - 2 * conv_transpose_padding + conv_transpose_output_padding
    
    # Create output tensor
    output = torch.empty((batch_size, out_channels, out_height, out_width), device=x.device, dtype=x.dtype)
    
    # Call fused kernel
    fused_ext.fused_conv_transpose_activation(
        x,
        conv_transpose_weight,
        conv_transpose_bias,
        float(add_value),
        float(multiply_value),
        output,
        kernel_size,
        conv_transpose_stride,
        conv_transpose_padding,
        conv_transpose_output_padding,
        conv_transpose_groups,
        conv_transpose_dilation
    )
    
    return output

# Test parameters
batch_size = 128
in_channels = 64
out_channels = 128
height, width = 64, 64
kernel_size = 4
stride = 2
add_value = 0.5
multiply_value = 2.0

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, add_value, multiply_value]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]
