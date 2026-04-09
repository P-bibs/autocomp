# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_052229/code_0.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_weight', 'conv_bias', 'conv_stride', 'conv_padding', 'conv_dilation', 'conv_groups']
REQUIRED_FLAT_STATE_NAMES = ['conv_weight', 'conv_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a convolution, applies HardSwish, and then ReLU.
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

# CUDA kernel for fused convolution + hardswish + relu
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_conv_activation_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    int groups,
    int out_height,
    int out_width
) {
    int batch_idx = blockIdx.x;
    int out_channel = blockIdx.y;
    int out_y = blockIdx.z;
    int out_x = threadIdx.x + blockDim.x * threadIdx.y;
    
    if (batch_idx >= batch_size || out_channel >= out_channels || out_y >= out_height || out_x >= out_width) return;
    
    float sum = 0.0f;
    
    // Convolution computation
    for (int in_c = 0; in_c < in_channels; in_c++) {
        for (int k_y = 0; k_y < kernel_size; k_y++) {
            for (int k_x = 0; k_x < kernel_size; k_x++) {
                int in_y = out_y * stride - padding + k_y * dilation;
                int in_x = out_x * stride - padding + k_x * dilation;
                
                if (in_y >= 0 && in_y < height && in_x >= 0 && in_x < width) {
                    int input_idx = batch_idx * (in_channels * height * width) + 
                                  in_c * (height * width) + 
                                  in_y * width + in_x;
                    int weight_idx = out_channel * (in_channels * kernel_size * kernel_size) + 
                                   in_c * (kernel_size * kernel_size) + 
                                   k_y * kernel_size + k_x;
                    
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    // Add bias
    sum += bias[out_channel];
    
    // Apply hardswish: x * relu6(x + 3) / 6
    float hardswish_val = sum * fminf(fmaxf(sum + 3.0f, 0.0f), 6.0f) / 6.0f;
    
    // Apply relu
    float relu_val = fmaxf(hardswish_val, 0.0f);
    
    // Write output
    int output_idx = batch_idx * (out_channels * out_height * out_width) + 
                   out_channel * (out_height * out_width) + 
                   out_y * out_width + out_x;
    output[output_idx] = relu_val;
}

void fused_conv_activation_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int stride,
    int padding,
    int dilation,
    int groups
) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    auto out_channels = weight.size(0);
    auto kernel_size = weight.size(2);
    
    int out_height = (height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_width = (width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    // Launch configuration
    dim3 grid(batch_size, out_channels, out_height);
    dim3 block(min(out_width, 32), min(32, max(1, 1024/(min(out_width, 32)))), 1);
    
    fused_conv_activation_kernel<<<grid, block>>>(
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
        dilation,
        groups,
        out_height,
        out_width
    );
    
    cudaDeviceSynchronize();
}
"""

# C++ binding code
cpp_source = r"""
#include <torch/extension.h>

void fused_conv_activation_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int stride,
    int padding,
    int dilation,
    int groups
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_activation", &fused_conv_activation_forward, "Fused convolution with activation functions");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_conv_activation',
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
    # Create output tensor with correct shape
    out_channels = conv_weight.size(0)
    in_height, in_width = x.size(2), x.size(3)
    kernel_size = conv_weight.size(2)
    
    out_height = (in_height + 2 * conv_padding - conv_dilation * (kernel_size - 1) - 1) // conv_stride + 1
    out_width = (in_width + 2 * conv_padding - conv_dilation * (kernel_size - 1) - 1) // conv_stride + 1
    
    output = torch.empty(x.size(0), out_channels, out_height, out_width, device=x.device, dtype=x.dtype)
    
    # Call the fused CUDA kernel
    fused_ext.fused_conv_activation(
        x, conv_weight, conv_bias, output,
        conv_stride, conv_padding, conv_dilation, conv_groups
    )
    
    return output

batch_size = 128
in_channels = 8
out_channels = 64
height, width = 128, 128
kernel_size = 3

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]
