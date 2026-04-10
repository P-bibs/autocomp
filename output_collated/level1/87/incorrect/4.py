# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_065607/code_2.py
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused convolution operation
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

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
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w
) {
    // Calculate output dimensions
    int out_h = (height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    int out_w = (width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = batch_size * out_channels * out_h * out_w;
    
    if (tid >= total_threads) return;
    
    // Calculate indices
    int w_idx = tid % out_w;
    int h_idx = (tid / out_w) % out_h;
    int c_idx = (tid / (out_w * out_h)) % out_channels;
    int b_idx = tid / (out_w * out_h * out_channels);
    
    // Calculate input position
    int in_h_start = h_idx * stride_h - padding_h;
    int in_w_start = w_idx * stride_w - padding_w;
    
    float sum = 0.0f;
    
    // Perform convolution
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int in_h = in_h_start + kh * dilation_h;
                int in_w = in_w_start + kw * dilation_w;
                
                float val = 0.0f;
                if (in_h >= 0 && in_h < height && in_w >= 0 && in_w < width) {
                    val = input[((b_idx * in_channels + ic) * height + in_h) * width + in_w];
                }
                
                sum += val * weight[((c_idx * in_channels + ic) * kernel_h + kh) * kernel_w + kw];
            }
        }
    }
    
    // Add bias
    sum += bias[c_idx];
    
    // Write output
    output[(((b_idx * out_channels + c_idx) * out_h + h_idx) * out_w + w_idx)] = sum;
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
    // Set device guard
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    int out_channels = weight.size(0);
    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);
    
    // Calculate output dimensions
    int out_h = (height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    int out_w = (width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
    
    // Validate output tensor size
    TORCH_CHECK(output.size(0) == batch_size);
    TORCH_CHECK(output.size(1) == out_channels);
    TORCH_CHECK(output.size(2) == out_h);
    TORCH_CHECK(output.size(3) == out_w);
    
    int total_threads = batch_size * out_channels * out_h * out_w;
    int threads_per_block = 256;
    int num_blocks = (total_threads + threads_per_block - 1) / threads_per_block;
    
    // Launch kernel
    conv2d_kernel<<<num_blocks, threads_per_block>>>(
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
    
    cudaDeviceSynchronize();
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
    m.def("conv2d_forward", &conv2d_forward, "2D Convolution Forward Pass");
}
"""

# Compile the extension
conv_ext = load_inline(
    name='conv_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

batch_size = 16
in_channels = 64
out_channels = 128
width = 1024
height = 1024

# Global variables to store convolution parameters
conv_params = {
    'weight': None,
    'bias': None,
    'stride': (1, 1),
    'padding': (0, 0),
    'dilation': (1, 1),
    'groups': 1
}

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
    # Store parameters for later use
    global conv_params
    conv_params['weight'] = conv1d_weight
    conv_params['bias'] = conv1d_bias
    conv_params['stride'] = conv1d_stride if isinstance(conv1d_stride, (tuple, list)) else (conv1d_stride, conv1d_stride)
    conv_params['padding'] = conv1d_padding if isinstance(conv1d_padding, (tuple, list)) else (conv1d_padding, conv1d_padding)
    conv_params['dilation'] = conv1d_dilation if isinstance(conv1d_dilation, (tuple, list)) else (conv1d_dilation, conv1d_dilation)
    conv_params['groups'] = conv1d_groups
    
    # Validate groups (only supports groups=1 for this implementation)
    if conv1d_groups != 1:
        raise ValueError("Only groups=1 is supported in this custom implementation")
    
    # Create output tensor
    stride_h, stride_w = conv_params['stride']
    padding_h, padding_w = conv_params['padding']
    dilation_h, dilation_w = conv_params['dilation']
    
    kernel_h, kernel_w = conv1d_weight.shape[2], conv1d_weight.shape[3]
    out_h = (x.shape[2] + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
    out_w = (x.shape[3] + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1
    
    output = torch.empty((x.shape[0], conv1d_weight.shape[0], out_h, out_w), device=x.device, dtype=x.dtype)
    
    # Call custom CUDA kernel
    conv_ext.conv2d_forward(
        x, conv1d_weight, conv1d_bias, output,
        stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w
    )
    
    return output

def get_init_inputs():
    return [in_channels, out_channels]

def get_inputs():
    x = torch.rand(batch_size, in_channels, height, width, device='cuda')
    return [x]
