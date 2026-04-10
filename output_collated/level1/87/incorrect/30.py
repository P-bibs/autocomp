# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_072125/code_0.py
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

# Custom CUDA kernel for optimized convolution using implicit GEMM
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void implicit_gemm_conv2d_kernel(
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
    int pad_h,
    int pad_w,
    int stride_h,
    int stride_w,
    int dilation_h,
    int dilation_w,
    int out_height,
    int out_width
) {
    // Calculate output position
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = batch_size * out_channels * out_height * out_width;
    
    if (tid >= total_outputs) return;
    
    // Decompose linear index into 4D coordinates
    int w_out = tid % out_width;
    int h_out = (tid / out_width) % out_height;
    int c_out = (tid / (out_width * out_height)) % out_channels;
    int n = tid / (out_width * out_height * out_channels);
    
    float sum = 0.0f;
    
    // Convolution loop
    for (int c_in = 0; c_in < in_channels; c_in++) {
        for (int kh = 0; kh < kernel_h; kh++) {
            for (int kw = 0; kw < kernel_w; kw++) {
                int h_in = h_out * stride_h - pad_h + kh * dilation_h;
                int w_in = w_out * stride_w - pad_w + kw * dilation_w;
                
                if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                    int input_idx = n * (in_channels * height * width) + 
                                   c_in * (height * width) + 
                                   h_in * width + w_in;
                                   
                    int weight_idx = c_out * (in_channels * kernel_h * kernel_w) + 
                                    c_in * (kernel_h * kernel_w) + 
                                    kh * kernel_w + kw;
                                    
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    // Add bias
    if (bias != nullptr) {
        sum += bias[c_out];
    }
    
    output[tid] = sum;
}

void implicit_gemm_conv2d_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w
) {
    auto input_sizes = input.sizes();
    auto weight_sizes = weight.sizes();
    
    int batch_size = input_sizes[0];
    int in_channels = input_sizes[1];
    int height = input_sizes[2];
    int width = input_sizes[3];
    int out_channels = weight_sizes[0];
    int kernel_h = weight_sizes[2];
    int kernel_w = weight_sizes[3];
    
    int out_height = (height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    int out_width = (width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
    
    int total_threads = batch_size * out_channels * out_height * out_width;
    int threads_per_block = 256;
    int num_blocks = (total_threads + threads_per_block - 1) / threads_per_block;
    
    implicit_gemm_conv2d_kernel<<<num_blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        kernel_h,
        kernel_w,
        pad_h, pad_w,
        stride_h, stride_w,
        dilation_h, dilation_w,
        out_height,
        out_width
    );
}
"""

# --- C++ Logic (Interface/Bindings) ---
cpp_source = r"""
#include <torch/extension.h>

// Forward declaration of the function in the .cu file
void implicit_gemm_conv2d_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("implicit_gemm_conv2d", &implicit_gemm_conv2d_forward, "Implicit GEMM Conv2D forward");
}
"""

# Compile the extension
implicit_gemm_ext = load_inline(
    name='implicit_gemm_conv2d',
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
    # Handle stride, padding, dilation as tuples or single values
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
    
    # Ensure tensors are contiguous and on the right device
    x = x.contiguous()
    conv1d_weight = conv1d_weight.contiguous()
    
    # Create output tensor
    out_channels = conv1d_weight.size(0)
    in_height, in_width = x.size(2), x.size(3)
    
    out_height = (in_height + 2 * pad_h - dilation_h * (conv1d_weight.size(2) - 1) - 1) // stride_h + 1
    out_width = (in_width + 2 * pad_w - dilation_w * (conv1d_weight.size(3) - 1) - 1) // stride_w + 1
    
    output = torch.empty(x.size(0), out_channels, out_height, out_width, 
                        dtype=x.dtype, device=x.device)
    
    # Call optimized CUDA kernel
    implicit_gemm_ext.implicit_gemm_conv2d(
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
