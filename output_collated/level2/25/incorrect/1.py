# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_081239/code_0.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_weight', 'conv_bias', 'conv_stride', 'conv_padding', 'conv_dilation', 'conv_groups']
REQUIRED_FLAT_STATE_NAMES = ['conv_weight', 'conv_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a convolution, applies minimum operation, Tanh, and another Tanh.
    """

    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

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
    # State for conv (nn.Conv2d)
    if 'conv_weight' in flat_state:
        state_kwargs['conv_weight'] = flat_state['conv_weight']
    else:
        state_kwargs['conv_weight'] = getattr(model.conv, 'weight', None)
    if 'conv_bias' in flat_state:
        state_kwargs['conv_bias'] = flat_state['conv_bias']
    else:
        state_kwargs['conv_bias'] = getattr(model.conv, 'bias', None)
    state_kwargs['conv_stride'] = model.conv.stride
    state_kwargs['conv_padding'] = model.conv.padding
    state_kwargs['conv_dilation'] = model.conv.dilation
    state_kwargs['conv_groups'] = model.conv.groups
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

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_conv_min_tanh_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int kernel_size,
    int stride,
    int padding,
    int dilation) {
    
    int batch_idx = blockIdx.x;
    int out_ch = blockIdx.y;
    int spatial_idx = blockIdx.z * blockDim.x + threadIdx.x;
    
    int out_height = (height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_width = (width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    if (batch_idx >= batch_size || out_ch >= out_channels || spatial_idx >= out_height * out_width) {
        return;
    }
    
    int out_y = spatial_idx / out_width;
    int out_x = spatial_idx % out_width;
    
    float sum = (bias != nullptr) ? bias[out_ch] : 0.0f;
    
    // Convolution loop
    for (int in_ch = 0; in_ch < in_channels; ++in_ch) {
        for (int ky = 0; ky < kernel_size; ++ky) {
            for (int kx = 0; kx < kernel_size; ++kx) {
                int in_y = out_y * stride - padding + ky * dilation;
                int in_x = out_x * stride - padding + kx * dilation;
                
                if (in_y >= 0 && in_y < height && in_x >= 0 && in_x < width) {
                    int input_idx = ((batch_idx * in_channels) + in_ch) * height * width + in_y * width + in_x;
                    int weight_idx = (out_ch * in_channels + in_ch) * kernel_size * kernel_size + ky * kernel_size + kx;
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    // Apply tanh twice as in original code
    float result = tanhf(sum);
    result = tanhf(result);
    
    int output_idx = ((batch_idx * out_channels) + out_ch) * out_height * out_width + out_y * out_width + out_x;
    output[output_idx] = result;
}

// Kernel for min reduction across channels
__global__ void channel_min_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width) {
    
    int batch_idx = blockIdx.x;
    int spatial_idx = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (batch_idx >= batch_size || spatial_idx >= height * width) {
        return;
    }
    
    int y = spatial_idx / width;
    int x = spatial_idx % width;
    
    // Find minimum across channels
    float min_val = INFINITY;
    for (int ch = 0; ch < in_channels; ++ch) {
        int idx = ((batch_idx * in_channels) + ch) * height * width + y * width + x;
        min_val = fminf(min_val, input[idx]);
    }
    
    int out_idx = batch_idx * height * width + y * width + x;
    output[out_idx] = min_val;
}

void launch_fused_conv_tanh(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    int stride,
    int padding,
    int dilation) {
    
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);
    
    dim3 block_size(256);
    dim3 grid_size(batch_size, out_channels, 
                   ((height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1) * 
                   ((width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1) / 256 + 1);
    
    fused_conv_min_tanh_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        kernel_size,
        stride,
        padding,
        dilation
    );
}

void launch_channel_min(
    const torch::Tensor& input,
    torch::Tensor& output) {
    
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    
    dim3 block_size(256);
    dim3 grid_size(batch_size, (height * width + 255) / 256);
    
    channel_min_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        1,  // out_channels
        height,
        width
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void launch_fused_conv_tanh(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    int stride,
    int padding,
    int dilation);

void launch_channel_min(
    const torch::Tensor& input,
    torch::Tensor& output);

torch::Tensor fused_conv_tanh_op(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    int64_t stride,
    int64_t padding,
    int64_t dilation) {
    
    auto output_height = (input.size(2) + 2 * padding - dilation * (weight.size(2) - 1) - 1) / stride + 1;
    auto output_width = (input.size(3) + 2 * padding - dilation * (weight.size(3) - 1) - 1) / stride + 1;
    
    auto output = torch::zeros({input.size(0), weight.size(0), output_height, output_width}, 
                               torch::TensorOptions().dtype(input.dtype()).device(input.device()));
    
    launch_fused_conv_tanh(input, weight, bias, output, stride, padding, dilation);
    
    return output;
}

torch::Tensor channel_min_op(const torch::Tensor& input) {
    auto output = torch::zeros({input.size(0), 1, input.size(2), input.size(3)}, 
                               torch::TensorOptions().dtype(input.dtype()).device(input.device()));
    
    launch_channel_min(input, output);
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_tanh", &fused_conv_tanh_op, "Fused Conv + Tanh operation");
    m.def("channel_min", &channel_min_op, "Channel-wise minimum operation");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(
    x,
    *,
    conv_weight,
    conv_bias,
    conv_stride,
    conv_padding,
    conv_dilation,
    conv_groups,
):
    # Apply fused convolution + double tanh
    x = fused_ext.fused_conv_tanh(
        x.contiguous(),
        conv_weight.contiguous(),
        conv_bias.contiguous(),
        conv_stride,
        conv_padding,
        conv_dilation
    )
    
    # Apply min reduction across channels
    x = fused_ext.channel_min(x.contiguous())
    
    return x

batch_size = 128
in_channels = 16
out_channels = 64
height = width = 256
kernel_size = 3

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]
