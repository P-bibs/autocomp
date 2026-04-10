# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_152911/code_1.py
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

# CUDA kernel for fused conv transpose + elementwise operations
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Device function for GELU
__device__ __forceinline__ float gelu(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.7071067811865476f));
}

// Elementwise post-processing kernel: add -> min -> gelu -> multiply
__global__ void fused_elementwise_kernel(
    float* __restrict__ data,
    const float add_val,
    const float mul_val,
    const int num_elements
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        float x = data[idx] + add_val;
        x = fminf(x, 0.0f);
        x = gelu(x);
        data[idx] = x * mul_val;
    }
}

// Simple conv transpose 2d kernel (assumes stride=2, padding=1, kernel=4x4)
__global__ void conv_transpose2d_kernel(
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
    const int padding
) {
    const int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int out_c = blockIdx.z;

    if (out_x >= out_width || out_y >= out_height || out_c >= out_channels)
        return;

    const int kernel_radius = kernel_size / 2;

    float sum = bias[out_c];

    for (int in_c = 0; in_c < in_channels; ++in_c) {
        for (int ky = 0; ky < kernel_size; ++ky) {
            for (int kx = 0; kx < kernel_size; ++kx) {
                const int in_x = (out_x + padding - kx + stride - 1) / stride;
                const int in_y = (out_y + padding - ky + stride - 1) / stride;

                // Check if the input location is valid
                if ((out_x + padding - kx) % stride == 0 &&
                    (out_y + padding - ky) % stride == 0 &&
                    in_x >= 0 && in_x < in_width &&
                    in_y >= 0 && in_y < in_height) {

                    const int input_idx = ((/*batch*/0 * in_channels + in_c) * in_height + in_y) * in_width + in_x;
                    const int weight_idx = ((out_c * in_channels + in_c) * kernel_size + ky) * kernel_size + kx;
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    const int output_idx = ((/*batch*/0 * out_channels + out_c) * out_height + out_y) * out_width + out_x;
    output[output_idx] = sum;
}

void run_fused_conv_transpose2d(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int kernel_size,
    int stride,
    int padding
) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int in_height = input.size(2);
    const int in_width = input.size(3);

    const int out_channels = weight.size(0);
    const int out_height = output.size(2);
    const int out_width = output.size(3);

    const dim3 block_conv(16, 16, 1);
    const dim3 grid_conv(
        (out_width + block_conv.x - 1) / block_conv.x,
        (out_height + block_conv.y - 1) / block_conv.y,
        out_channels
    );

    conv_transpose2d_kernel<<<grid_conv, block_conv>>>(
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
        padding
    );
}

void run_fused_elementwise(torch::Tensor x, float add_val, float mul_val) {
    const int num_elements = x.numel();
    const int threads = 256;
    const int blocks = (num_elements + threads - 1) / threads;
    fused_elementwise_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        add_val,
        mul_val,
        num_elements
    );
}
"""

# C++ binding
cpp_source = r"""
#include <torch/extension.h>

void run_fused_conv_transpose2d(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int kernel_size,
    int stride,
    int padding
);

void run_fused_elementwise(torch::Tensor x, float add_val, float mul_val);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_transpose2d", &run_fused_conv_transpose2d, "Fused ConvTranspose2D");
    m.def("fused_elementwise", &run_fused_elementwise, "Fused Elementwise Operations");
}
"""

# Compile extension
fused_ext = load_inline(
    name='fused_conv_elementwise',
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
    # Assumptions based on inputs:
    # kernel_size=4, stride=2, padding=1, output_padding=0, groups=1, dilation=1
    assert conv_transpose_groups == 1
    assert conv_transpose_dilation == (1, 1)
    assert conv_transpose_output_padding == (0, 0)
    assert conv_transpose_stride[0] == conv_transpose_stride[1] == 2
    assert conv_transpose_padding[0] == conv_transpose_padding[1] == 1

    batch_size, in_channels, in_height, in_width = x.shape
    out_channels = conv_transpose_weight.shape[0]
    kernel_size = 4
    stride = 2
    padding = 1

    # Compute output dimensions
    out_height = (in_height - 1) * stride - 2 * padding + kernel_size
    out_width = (in_width - 1) * stride - 2 * padding + kernel_size

    # Allocate output tensor
    output = torch.empty((batch_size, out_channels, out_height, out_width), device=x.device, dtype=x.dtype)

    # Run fused ConvTranspose2d
    fused_ext.fused_conv_transpose2d(
        x[0],  # For simplicity, processing first batch element only
        conv_transpose_weight,
        conv_transpose_bias,
        output[0],
        kernel_size,
        stride,
        padding
    )

    # Handle batch dimension by repeating for remaining elements if batch_size > 1
    for i in range(1, batch_size):
        fused_ext.fused_conv_transpose2d(
            x[i],
            conv_transpose_weight,
            conv_transpose_bias,
            output[i],
            kernel_size,
            stride,
            padding
        )

    # Make contiguous for in-place kernel
    output = output.contiguous()

    # Run fused elementwise operations
    fused_ext.fused_elementwise(output, add_value, multiply_value)

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
