# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_072125/code_2.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'bias']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv1d_weight', 'conv1d_bias', 'conv1d_stride', 'conv1d_padding', 'conv1d_dilation', 'conv1d_groups']
REQUIRED_FLAT_STATE_NAMES = ['conv1d_weight', 'conv1d_bias']


class ModelNew(nn.Module):
    """
    Performs a pointwise 2D convolution operation.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """

    def __init__(self, in_channels: int, out_channels: int, bias: bool=False):
        super(ModelNew, self).__init__()
        self.conv1d = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)

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
    # State for conv1d (nn.Conv2d)
    if 'conv1d_weight' in flat_state:
        state_kwargs['conv1d_weight'] = flat_state['conv1d_weight']
    else:
        state_kwargs['conv1d_weight'] = getattr(model.conv1d, 'weight', None)
    if 'conv1d_bias' in flat_state:
        state_kwargs['conv1d_bias'] = flat_state['conv1d_bias']
    else:
        state_kwargs['conv1d_bias'] = getattr(model.conv1d, 'bias', None)
    state_kwargs['conv1d_stride'] = model.conv1d.stride
    state_kwargs['conv1d_padding'] = model.conv1d.padding
    state_kwargs['conv1d_dilation'] = model.conv1d.dilation
    state_kwargs['conv1d_groups'] = model.conv1d.groups
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

# CUDA kernel for fused 2D convolution
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int kernel_h,
    int kernel_w,
    int out_height,
    int out_width,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w
) {
    // Calculate global thread index
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_ch = blockIdx.z;
    
    // Check bounds
    if (out_x >= out_width || out_y >= out_height || out_ch >= out_channels) return;
    
    float sum = (bias != nullptr) ? bias[out_ch] : 0.0f;
    
    // Perform convolution
    for (int in_ch = 0; in_ch < in_channels; ++in_ch) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int in_y = out_y * stride_h - pad_h + kh * dilation_h;
                int in_x = out_x * stride_w - pad_w + kw * dilation_w;
                
                if (in_y >= 0 && in_y < height && in_x >= 0 && in_x < width) {
                    int in_idx = ((blockIdx.z * in_channels + in_ch) * height + in_y) * width + in_x;
                    int w_idx = (out_ch * in_channels + in_ch) * kernel_h * kernel_w + kh * kernel_w + kw;
                    sum += input[in_idx] * weight[w_idx];
                }
            }
        }
    }
    
    int out_idx = ((blockIdx.z * out_height + out_y) * out_width + out_x);
    output[out_idx] = sum;
}

void conv2d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    int out_channels = weight.size(0);
    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);
    
    // Calculate output dimensions
    int out_height = (height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    int out_width = (width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
    
    // Define block and grid dimensions
    dim3 block(16, 16, 1);
    dim3 grid(
        (out_width + block.x - 1) / block.x,
        (out_height + block.y - 1) / block.y,
        out_channels
    );
    
    // Launch kernel
    conv2d_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        kernel_h,
        kernel_w,
        out_height,
        out_width,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void conv2d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv2d", &conv2d_forward, "2D Convolution CUDA Kernel");
}
"""

# Compile CUDA extension
conv_ext = load_inline(
    name='conv_ops',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math', '-arch=sm_75'],
    with_cuda=True
)

def functional_model(
    x,
    *,
    conv1d_weight,
    conv1d_bias,
    conv1d_stride,
    conv1d_padding,
    conv1d_dilation,
    conv1d_groups,
):
    # Handle stride, padding, dilation as either single int or tuple
    if isinstance(conv1d_stride, int):
        stride_h = stride_w = conv1d_stride
    else:
        stride_h, stride_w = conv1d_stride
    
    if isinstance(conv1d_padding, int):
        pad_h = pad_w = conv1d_padding
    else:
        pad_h, pad_w = conv1d_padding
        
    if isinstance(conv1d_dilation, int):
        dilation_h = dilation_w = conv1d_dilation
    else:
        dilation_h, dilation_w = conv1d_dilation
    
    # Calculate output dimensions
    batch_size = x.size(0)
    in_channels = x.size(1)
    height = x.size(2)
    width = x.size(3)
    out_channels = conv1d_weight.size(0)
    kernel_h = conv1d_weight.size(2)
    kernel_w = conv1d_weight.size(3)
    
    out_height = (height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
    out_width = (width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1
    
    # Allocate output tensor
    output = torch.empty(batch_size, out_channels, out_height, out_width, 
                        dtype=x.dtype, device=x.device)
    
    # Call custom CUDA kernel
    conv_ext.conv2d(
        x, conv1d_weight, conv1d_bias, output,
        stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w
    )
    
    return output

batch_size = 16
in_channels = 64
out_channels = 128
width = 1024
height = 1024

def get_init_inputs():
    return [in_channels, out_channels]

def get_inputs():
    x = torch.rand(batch_size, in_channels, height, width)
    return [x]
