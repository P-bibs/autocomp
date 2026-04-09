# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_050905/code_2.py
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

# CUDA kernel for fused conv2d + hardswish + relu operation
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

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
    int groups
) {
    // Calculate output dimensions
    int out_height = (height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_width = (width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * out_height * out_width;
    
    if (tid >= total_elements) return;
    
    // Calculate indices
    int tmp = tid;
    int out_w = tmp % out_width;
    tmp /= out_width;
    int out_h = tmp % out_height;
    tmp /= out_height;
    int out_c = tmp % out_channels;
    int batch = tmp / out_channels;
    
    // Calculate convolution value
    float sum = 0.0f;
    
    // For grouped convolutions, determine which group this output channel belongs to
    int group_id = out_c * groups / out_channels;
    int in_channels_per_group = in_channels / groups;
    int out_channels_per_group = out_channels / groups;
    int weight_offset_base = group_id * in_channels_per_group * out_channels_per_group * kernel_size * kernel_size;
    
    for (int in_c_group = 0; in_c_group < in_channels_per_group; in_c_group++) {
        int in_c_global = group_id * in_channels_per_group + in_c_group;
        int weight_c_out = (out_c % out_channels_per_group);
        
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                int in_h = out_h * stride - padding + kh * dilation;
                int in_w = out_w * stride - padding + kw * dilation;
                
                if (in_h >= 0 && in_h < height && in_w >= 0 && in_w < width) {
                    int input_idx = batch * (in_channels * height * width) + 
                                   in_c_global * (height * width) + 
                                   in_h * width + in_w;
                                   
                    int weight_idx = weight_offset_base + 
                                    weight_c_out * (in_channels_per_group * kernel_size * kernel_size) +
                                    in_c_group * (kernel_size * kernel_size) +
                                    kh * kernel_size + kw;
                                    
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    // Add bias
    sum += bias[out_c];
    
    // Apply hardswish: x * relu6(x+3) / 6
    float hardswish_val;
    if (sum <= -3.0f) {
        hardswish_val = 0.0f;
    } else if (sum >= 3.0f) {
        hardswish_val = sum;
    } else {
        hardswish_val = sum * (sum + 3.0f) / 6.0f;
    }
    
    // Apply relu
    float result = fmaxf(hardswish_val, 0.0f);
    
    output[tid] = result;
}

void fused_conv_activation_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    int groups
) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    int out_channels = weight.size(0);
    
    int out_height = (height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_width = (width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    int total_elements = batch_size * out_channels * out_height * out_width;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    
    fused_conv_activation_kernel<<<blocks, threads>>>(
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
        groups
    );
    
    cudaDeviceSynchronize();
}
"""

# C++ interface for the CUDA kernel
cpp_source = r"""
#include <torch/extension.h>

void fused_conv_activation_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    int groups
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_activation", &fused_conv_activation_forward, "Fused Conv2d + Hardswish + ReLU");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_conv_activation_ext',
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
    batch_size = x.size(0)
    in_channels = x.size(1)
    height = x.size(2)
    width = x.size(3)
    out_channels = conv_weight.size(0)
    kernel_size = conv_weight.size(2)
    
    # Calculate output dimensions
    out_height = (height + 2 * conv_padding - conv_dilation * (kernel_size - 1) - 1) // conv_stride + 1
    out_width = (width + 2 * conv_padding - conv_dilation * (kernel_size - 1) - 1) // conv_stride + 1
    
    # Create output tensor
    output = torch.empty((batch_size, out_channels, out_height, out_width), device=x.device, dtype=x.dtype)
    
    # Call fused kernel
    fused_ext.fused_conv_activation(
        x,
        conv_weight,
        conv_bias,
        output,
        kernel_size,
        conv_stride,
        conv_padding,
        conv_dilation,
        conv_groups
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
