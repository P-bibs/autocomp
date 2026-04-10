# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_155151/code_5.py
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused elementwise operations
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__device__ float gelu(const float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

__global__ void fused_elementwise_ops_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float add_value,
    const float multiply_value,
    const int num_elements
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        float val = input[idx];
        val += add_value;
        val = fminf(val, 0.0f);
        val = gelu(val);
        val *= multiply_value;
        output[idx] = val;
    }
}

void launch_fused_elementwise_ops(
    const torch::Tensor& input,
    torch::Tensor& output,
    const float add_value,
    const float multiply_value
) {
    const int num_elements = input.numel();
    const int threads_per_block = 256;
    const int blocks = (num_elements + threads_per_block - 1) / threads_per_block;
    
    fused_elementwise_ops_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        add_value,
        multiply_value,
        num_elements
    );
}
"""

# Custom CUDA kernel for transposed convolution
conv_transpose_cuda = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

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
    const int kernel_size,
    const int stride,
    const int padding,
    const int output_padding,
    const int groups,
    const int dilation
) {
    const int out_height = (in_height - 1) * stride - 2 * padding + kernel_size + output_padding;
    const int out_width = (in_width - 1) * stride - 2 * padding + kernel_size + output_padding;
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * out_channels * out_height * out_width;
    
    if (tid >= total_elements) return;
    
    const int n = tid / (out_channels * out_height * out_width);
    const int c_out = (tid / (out_height * out_width)) % out_channels;
    const int h_out = (tid / out_width) % out_height;
    const int w_out = tid % out_width;
    
    float sum = (bias != nullptr) ? bias[c_out] : 0.0f;
    
    const int group = c_out * groups / out_channels;
    const int in_channels_per_group = in_channels / groups;
    
    for (int c_in_group = 0; c_in_group < in_channels_per_group; c_in_group++) {
        const int c_in = group * in_channels_per_group + c_in_group;
        
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                const int h_in = h_out + padding - kh * dilation;
                const int w_in = w_out + padding - kw * dilation;
                
                if (h_in % stride == 0 && w_in % stride == 0) {
                    const int h_in_pixel = h_in / stride;
                    const int w_in_pixel = w_in / stride;
                    
                    if (h_in_pixel >= 0 && h_in_pixel < in_height && 
                        w_in_pixel >= 0 && w_in_pixel < in_width) {
                        
                        const int input_idx = ((n * in_channels + c_in) * in_height + h_in_pixel) * in_width + w_in_pixel;
                        const int weight_idx = ((c_out * in_channels_per_group + c_in_group) * kernel_size + kh) * kernel_size + kw;
                        
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }
    
    output[tid] = sum;
}

void launch_conv_transpose2d(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    const int kernel_size,
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
    
    const int out_height = (in_height - 1) * stride - 2 * padding + kernel_size + output_padding;
    const int out_width = (in_width - 1) * stride - 2 * padding + kernel_size + output_padding;
    
    const int total_elements = batch_size * out_channels * out_height * out_width;
    const int threads_per_block = 256;
    const int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    conv_transpose2d_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        in_height,
        in_width,
        kernel_size,
        stride,
        padding,
        output_padding,
        groups,
        dilation
    );
}
"""

# C++ bindings
cpp_source = r"""
#include <torch/extension.h>

void launch_fused_elementwise_ops(
    const torch::Tensor& input,
    torch::Tensor& output,
    const float add_value,
    const float multiply_value
);

void launch_conv_transpose2d(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    const int kernel_size,
    const int stride,
    const int padding,
    const int output_padding,
    const int groups,
    const int dilation
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_elementwise_ops", &launch_fused_elementwise_ops, "Fused elementwise operations");
    m.def("conv_transpose2d", &launch_conv_transpose2d, "Custom conv transpose 2d");
}
"""

# Compile the extensions
fused_ext = load_inline(
    name='fused_ops',
    cpp_sources=cpp_source,
    cuda_sources=[cuda_kernel, conv_transpose_cuda],
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
    # Calculate output dimensions for conv transpose
    in_height, in_width = x.shape[2], x.shape[3]
    kernel_size = conv_transpose_weight.shape[2]
    out_height = (in_height - 1) * conv_transpose_stride - 2 * conv_transpose_padding + kernel_size + conv_transpose_output_padding
    out_width = (in_width - 1) * conv_transpose_stride - 2 * conv_transpose_padding + kernel_size + conv_transpose_output_padding
    out_channels = conv_transpose_weight.shape[1]
    
    # Create output tensor for conv transpose
    conv_output = torch.empty((x.shape[0], out_channels, out_height, out_width), device=x.device, dtype=x.dtype)
    
    # Launch custom conv transpose kernel
    fused_ext.conv_transpose2d(
        x, conv_transpose_weight, conv_transpose_bias, conv_output,
        kernel_size, conv_transpose_stride, conv_transpose_padding,
        conv_transpose_output_padding, conv_transpose_groups, conv_transpose_dilation
    )
    
    # Create output tensor for fused elementwise operations
    final_output = torch.empty_like(conv_output)
    
    # Launch fused elementwise operations kernel
    fused_ext.fused_elementwise_ops(
        conv_output, final_output, add_value, multiply_value
    )
    
    return final_output

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
