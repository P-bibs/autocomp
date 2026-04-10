# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_071708/code_2.py
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
import math
from torch.utils.cpp_extension import load_inline

# Define the optimized CUDA kernel using Implicit GEMM approach
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

#define TILE_SIZE 16
#define BLOCK_SIZE 256

template <typename scalar_t>
__global__ void implicit_gemm_conv2d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w,
    int groups) {
    
    // Calculate output dimensions
    int out_h = (height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    int out_w = (width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
    
    // Shared memory for input and weight tiles
    __shared__ scalar_t input_tile[TILE_SIZE][TILE_SIZE];
    __shared__ scalar_t weight_tile[TILE_SIZE][TILE_SIZE];
    
    int batch_idx = blockIdx.x;
    int out_ch_idx = blockIdx.y;
    int thread_idx = threadIdx.x;
    
    if (batch_idx >= batch_size || out_ch_idx >= out_channels)
        return;
        
    // Each thread calculates one output element
    for (int oh = 0; oh < out_h; oh++) {
        for (int ow = 0; ow < out_w; ow++) {
            scalar_t sum = 0.0;
            
            // Perform convolution operation
            for (int ic = 0; ic < in_channels; ic++) {
                for (int kh = 0; kh < kernel_h; kh++) {
                    for (int kw = 0; kw < kernel_w; kw++) {
                        int ih = oh * stride_h - pad_h + kh * dilation_h;
                        int iw = ow * stride_w - pad_w + kw * dilation_w;
                        
                        scalar_t input_val = 0.0;
                        if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                            input_val = input[batch_idx * (in_channels * height * width) + 
                                            ic * (height * width) + 
                                            ih * width + iw];
                        }
                        
                        scalar_t weight_val = weight[out_ch_idx * (in_channels * kernel_h * kernel_w) + 
                                                   ic * (kernel_h * kernel_w) + 
                                                   kh * kernel_w + kw];
                        
                        sum += input_val * weight_val;
                    }
                }
            }
            
            // Add bias
            if (bias != nullptr) {
                sum += bias[out_ch_idx];
            }
            
            // Write output
            output[batch_idx * (out_channels * out_h * out_w) + 
                  out_ch_idx * (out_h * out_w) + 
                  oh * out_w + ow] = sum;
        }
    }
}

void implicit_gemm_conv2d(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w,
    int groups) {
    
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    auto out_channels = weight.size(0);
    auto kernel_h = weight.size(2);
    auto kernel_w = weight.size(3);
    
    dim3 grid(batch_size, out_channels);
    dim3 block(BLOCK_SIZE);
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "implicit_gemm_conv2d_kernel", ([&] {
        implicit_gemm_conv2d_kernel<scalar_t><<<grid, block>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.defined() ? bias.data_ptr<scalar_t>() : nullptr,
            output.data_ptr<scalar_t>(),
            batch_size,
            in_channels,
            out_channels,
            height,
            width,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            dilation_h,
            dilation_w,
            groups
        );
    }));
    
    cudaDeviceSynchronize();
}
"""

# C++ source for PyTorch binding
cpp_source = r"""
#include <torch/extension.h>

void implicit_gemm_conv2d(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w,
    int groups);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("implicit_gemm_conv2d", &implicit_gemm_conv2d, "Implicit GEMM Conv2D");
}
"""

# Compile the extension
implicit_gemm_ext = load_inline(
    name='implicit_gemm_conv2d_ext',
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
    # Convert stride, padding, dilation to tuples if they aren't already
    if isinstance(conv1d_stride, int):
        stride_h, stride_w = conv1d_stride, conv1d_stride
    else:
        stride_h, stride_w = conv1d_stride[0], conv1d_stride[1]
        
    if isinstance(conv1d_padding, int):
        pad_h, pad_w = conv1d_padding, conv1d_padding
    else:
        pad_h, pad_w = conv1d_padding[0], conv1d_padding[1]
        
    if isinstance(conv1d_dilation, int):
        dilation_h, dilation_w = conv1d_dilation, conv1d_dilation
    else:
        dilation_h, dilation_w = conv1d_dilation[0], conv1d_dilation[1]
    
    # Create output tensor with correct shape
    out_h = (x.size(2) + 2 * pad_h - dilation_h * (conv1d_weight.size(2) - 1) - 1) // stride_h + 1
    out_w = (x.size(3) + 2 * pad_w - dilation_w * (conv1d_weight.size(3) - 1) - 1) // stride_w + 1
    output = torch.empty(x.size(0), conv1d_weight.size(0), out_h, out_w, device=x.device, dtype=x.dtype)
    
    # Call the optimized CUDA kernel
    implicit_gemm_ext.implicit_gemm_conv2d(
        x, conv1d_weight, conv1d_bias, output,
        stride_h, stride_w,
        pad_h, pad_w,
        dilation_h, dilation_w,
        conv1d_groups
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
    x = torch.rand(batch_size, in_channels, height, width, device='cuda')
    return [x]
