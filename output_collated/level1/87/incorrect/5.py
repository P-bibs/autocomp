# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_065607/code_0.py
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

# CUDA kernel for fused convolution operation
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

__global__ void conv2d_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w
) {
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_ch = blockIdx.z;
    
    if (out_x >= (width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1 ||
        out_y >= (height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1 ||
        out_ch >= out_channels) {
        return;
    }
    
    float sum = 0.0f;
    
    for (int in_ch = 0; in_ch < in_channels; ++in_ch) {
        for (int ky = 0; ky < kernel_h; ++ky) {
            for (int kx = 0; kx < kernel_w; ++kx) {
                int iy = out_y * stride_h - padding_h + ky * dilation_h;
                int ix = out_x * stride_w - padding_w + kx * dilation_w;
                
                if (iy >= 0 && iy < height && ix >= 0 && ix < width) {
                    int input_idx = ((/*batch*/ 0 * in_channels + in_ch) * height + iy) * width + ix;
                    int weight_idx = ((out_ch * in_channels + in_ch) * kernel_h + ky) * kernel_w + kx;
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    int out_height = (height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    int out_width = (width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
    int output_idx = ((/*batch*/ 0 * out_channels + out_ch) * out_height + out_y) * out_width + out_x;
    output[output_idx] = sum + bias[out_ch];
}

void conv2d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w
) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    int out_channels = weight.size(0);
    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);
    
    int out_height = (height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    int out_width = (width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
    
    dim3 block_size(16, 16, 1);
    dim3 grid_size(
        (out_width + block_size.x - 1) / block_size.x,
        (out_height + block_size.y - 1) / block_size.y,
        out_channels
    );
    
    conv2d_kernel<<<grid_size, block_size>>>(
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
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w
    );
}
"""

# C++ binding code
cpp_source = r"""
#include <torch/extension.h>

void conv2d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv2d_forward", &conv2d_forward, "2D convolution forward pass");
}
"""

# Compile the custom CUDA extension
conv_ext = load_inline(
    name='conv_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
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
    # Move tensors to GPU if not already
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = x.to(device)
    conv1d_weight = conv1d_weight.to(device)
    conv1d_bias = conv1d_bias.to(device)
    
    # Extract stride, padding, dilation values
    stride_h, stride_w = conv1d_stride if isinstance(conv1d_stride, (tuple, list)) else (conv1d_stride, conv1d_stride)
    padding_h, padding_w = conv1d_padding if isinstance(conv1d_padding, (tuple, list)) else (conv1d_padding, conv1d_padding)
    dilation_h, dilation_w = conv1d_dilation if isinstance(conv1d_dilation, (tuple, list)) else (conv1d_dilation, conv1d_dilation)
    
    # Calculate output dimensions
    batch_size, in_channels, height, width = x.shape
    out_channels, _, kernel_h, kernel_w = conv1d_weight.shape
    
    out_height = (height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
    out_width = (width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, out_height, out_width, device=device, dtype=x.dtype)
    
    # Call custom CUDA kernel
    conv_ext.conv2d_forward(
        x, conv1d_weight, conv1d_bias, output,
        stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w
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
