# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_050613/code_2.py
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

# CUDA kernel for fused convolution + hardswish + relu operation
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__device__ float relu6(float x) {
    return fminf(fmaxf(x, 0.0f), 6.0f);
}

__device__ float hardswish(float x) {
    return x * relu6(x + 3.0f) / 6.0f;
}

__global__ void fused_conv_hardswish_relu_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int in_height,
    int in_width,
    int out_channels,
    int kernel_size,
    int out_height,
    int out_width,
    int stride,
    int padding,
    int dilation,
    int groups
) {
    int out_ch = blockIdx.x;
    int batch_idx = blockIdx.y;
    int out_y = threadIdx.y + blockIdx.z * blockDim.y;
    int out_x = threadIdx.x + blockIdx.z * blockDim.x;

    if (out_ch >= out_channels || batch_idx >= batch_size || out_y >= out_height || out_x >= out_width)
        return;

    float sum = 0.0f;
    
    // Calculate convolution
    int group_id = out_ch / (out_channels / groups);
    int in_ch_start = group_id * (in_channels / groups);
    int in_ch_end = in_ch_start + (in_channels / groups);
    
    for (int in_ch = in_ch_start; in_ch < in_ch_end; ++in_ch) {
        for (int ky = 0; ky < kernel_size; ++ky) {
            for (int kx = 0; kx < kernel_size; ++kx) {
                int in_y = out_y * stride - padding + ky * dilation;
                int in_x = out_x * stride - padding + kx * dilation;
                
                if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
                    float val = input[batch_idx * (in_channels * in_height * in_width) + 
                                     in_ch * (in_height * in_width) + 
                                     in_y * in_width + in_x];
                    float wgt = weight[out_ch * (in_channels * kernel_size * kernel_size) / groups + 
                                      (in_ch - in_ch_start) * (kernel_size * kernel_size) + 
                                      ky * kernel_size + kx];
                    sum += val * wgt;
                }
            }
        }
    }
    
    // Add bias
    sum += bias[out_ch];
    
    // Apply activations: hardswish then relu
    float result = hardswish(sum);
    result = fmaxf(result, 0.0f); // relu
    
    // Write output
    output[batch_idx * (out_channels * out_height * out_width) + 
           out_ch * (out_height * out_width) + 
           out_y * out_width + out_x] = result;
}
"""

# C++ interface
cpp_source = r"""
#include <torch/extension.h>

void fused_conv_hardswish_relu_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int in_height,
    int in_width,
    int out_channels,
    int kernel_size,
    int out_height,
    int out_width,
    int stride,
    int padding,
    int dilation,
    int groups
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_hardswish_relu", &fused_conv_hardswish_relu_kernel, "Fused Conv + Hardswish + ReLU forward");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_conv_hardswish_relu',
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
    batch_size, in_channels, in_height, in_width = x.shape
    out_channels, _, kernel_size, _ = conv_weight.shape
    
    # Calculate output dimensions
    out_height = (in_height + 2 * conv_padding - conv_dilation * (kernel_size - 1) - 1) // conv_stride + 1
    out_width = (in_width + 2 * conv_padding - conv_dilation * (kernel_size - 1) - 1) // conv_stride + 1
    
    # Allocate output tensor
    output = torch.empty(batch_size, out_channels, out_height, out_width, device=x.device, dtype=x.dtype)
    
    # Configure kernel launch parameters
    threads_per_block = (16, 16)
    blocks_per_grid = (
        out_channels,
        batch_size,
        ((out_height + threads_per_block[1] - 1) // threads_per_block[1]) * 
        ((out_width + threads_per_block[0] - 1) // threads_per_block[0])
    )
    
    # Launch kernel
    fused_ext.fused_conv_hardswish_relu(
        x.contiguous().data_ptr(torch.float32),
        conv_weight.contiguous().data_ptr(torch.float32),
        conv_bias.contiguous().data_ptr(torch.float32),
        output.data_ptr(torch.float32),
        batch_size,
        in_channels,
        in_height,
        in_width,
        out_channels,
        kernel_size,
        out_height,
        out_width,
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
