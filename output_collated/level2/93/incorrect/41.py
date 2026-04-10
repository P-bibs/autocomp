# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_155151/code_3.py
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
from torch.utils.cpp_extension import load_inline

# --- CUDA Kernel Code ---
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

#define GELU_CONSTANT 0.7978845608f
#define GELU_CONSTANT2 0.044715f

// CUDA kernel for fused ConvTranspose2d + pointwise operations
__global__ void fused_conv_transpose2d_gelu_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
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
    const float add_value,
    const float multiply_value
) {
    // Global thread index
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output_elements = batch_size * out_channels * out_height * out_width;

    if (out_idx >= total_output_elements) return;

    // Decompose output index into (batch, out_c, out_y, out_x)
    int batch_idx = out_idx / (out_channels * out_height * out_width);
    int remainder = out_idx % (out_channels * out_height * out_width);
    int out_c = remainder / (out_height * out_width);
    remainder = remainder % (out_height * out_width);
    int out_y = remainder / out_width;
    int out_x = remainder % out_width;

    // Calculate corresponding input region for this output position
    int start_in_y = (out_y + padding - kernel_size + 1 + stride - 1) / stride; // Ceil division
    int start_in_x = (out_x + padding - kernel_size + 1 + stride - 1) / stride;
    int end_in_y = min((out_y + padding) / stride + 1, in_height); // Floor division + 1
    int end_in_x = min((out_x + padding) / stride + 1, in_width);

    start_in_y = max(0, start_in_y);
    start_in_x = max(0, start_in_x);

    // Perform convolution transpose
    float conv_sum = 0.0f;

    // Loop over input feature map region
    for (int in_c = 0; in_c < in_channels; ++in_c) {
        for (int in_y = start_in_y; in_y < end_in_y; ++in_y) {
            for (int in_x = start_in_x; in_x < end_in_x; ++in_x) {
                // Determine kernel position
                int k_y = out_y - in_y * stride + padding;
                int k_x = out_x - in_x * stride + padding;

                // Check if kernel position is valid
                if (k_y >= 0 && k_y < kernel_size && k_x >= 0 && k_x < kernel_size) {
                    int input_idx = batch_idx * (in_channels * in_height * in_width) +
                                    in_c * (in_height * in_width) +
                                    in_y * in_width + in_x;

                    int weight_idx = out_c * (in_channels * kernel_size * kernel_size) +
                                     in_c * (kernel_size * kernel_size) +
                                     k_y * kernel_size + k_x;

                    conv_sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    // Add bias
    conv_sum += bias[out_c];

    // Fused pointwise operations: x = gelu(min(x + add_value, 0)) * multiply_value
    float x = conv_sum + add_value;
    x = fminf(x, 0.0f); // min(x, 0)
    
    // GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    float x_cubed = x * x * x;
    float tanh_input = GELU_CONSTANT * (x + GELU_CONSTANT2 * x_cubed);
    float gelu_result = 0.5f * x * (1.0f + tanhf(tanh_input));
    
    output[out_idx] = gelu_result * multiply_value;
}

void fused_conv_transpose2d_gelu(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::Tensor& output,
    const int kernel_size,
    const int stride,
    const int padding,
    const float add_value,
    const float multiply_value
) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int in_height = input.size(2);
    const int in_width = input.size(3);
    
    const int out_channels = weight.size(0);
    const int out_height = output.size(2);
    const int out_width = output.size(3);

    const int total_threads = batch_size * out_channels * out_height * out_width;
    const int threads_per_block = 256;
    const int num_blocks = (total_threads + threads_per_block - 1) / threads_per_block;

    fused_conv_transpose2d_gelu_kernel<<<num_blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
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
        add_value,
        multiply_value
    );
}
"""

# --- C++ Logic (Interface/Bindings) ---
cpp_source = r"""
#include <torch/extension.h>

void fused_conv_transpose2d_gelu(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::Tensor& output,
    const int kernel_size,
    const int stride,
    const int padding,
    const float add_value,
    const float multiply_value
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_transpose2d_gelu", &fused_conv_transpose2d_gelu, "Fused ConvTranspose2d + GELU");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_conv_transpose2d_gelu_ext',
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
    # Allocate output tensor
    batch_size = x.size(0)
    out_channels = conv_transpose_weight.size(0)
    in_height, in_width = x.size(2), x.size(3)
    
    # Compute output dimensions for transposed convolution
    out_height = (in_height - 1) * conv_transpose_stride + conv_transpose_dilation * (conv_transpose_weight.size(2) - 1) + 1 - 2 * conv_transpose_padding + conv_transpose_output_padding
    out_width = (in_width - 1) * conv_transpose_stride + conv_transpose_dilation * (conv_transpose_weight.size(3) - 1) + 1 - 2 * conv_transpose_padding + conv_transpose_output_padding
    
    out = torch.empty((batch_size, out_channels, out_height, out_width), device=x.device, dtype=x.dtype)
    
    # Launch fused CUDA kernel
    fused_ext.fused_conv_transpose2d_gelu(
        x,
        conv_transpose_weight,
        conv_transpose_bias,
        out,
        conv_transpose_weight.size(2),  # kernel_size (assuming square kernel)
        conv_transpose_stride,
        conv_transpose_padding,
        add_value,
        multiply_value
    )
    
    return out

# --- Configuration parameters ---
batch_size = 128
in_channels = 64
out_channels = 128
height, width = 64, 64
kernel_size = 4
stride = 2
add_value = 0.5
multiply_value = 2.0

# Helper functions for testing
def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, add_value, multiply_value]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]
