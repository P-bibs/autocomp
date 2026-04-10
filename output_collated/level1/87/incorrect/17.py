# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_071251/code_2.py
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

# CUDA kernel code
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
    int dilation_w,
    int groups,
    int out_height,
    int out_width
) {
    // Shared memory for weights
    extern __shared__ float shared_weights[];
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int batch_idx = blockIdx.z;
    const int out_ch_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (out_ch_idx >= out_channels) return;
    
    const int group_idx = out_ch_idx / (out_channels / groups);
    const int in_ch_per_group = in_channels / groups;
    const int weight_offset = group_idx * in_ch_per_group * kernel_h * kernel_w * out_channels / groups;
    
    // Load weights into shared memory
    const int weights_per_thread = (in_ch_per_group * kernel_h * kernel_w + blockDim.x - 1) / blockDim.x;
    for (int i = 0; i < weights_per_thread; ++i) {
        const int idx = tid + i * blockDim.x;
        if (idx < in_ch_per_group * kernel_h * kernel_w) {
            shared_weights[idx] = weight[weight_offset + out_ch_idx % (out_channels/groups) * in_ch_per_group * kernel_h * kernel_w + idx];
        }
    }
    __syncthreads();
    
    // Process output pixels
    const int pixels_per_thread = (out_height * out_width + blockDim.x - 1) / blockDim.x;
    for (int p = 0; p < pixels_per_thread; ++p) {
        const int pixel_idx = tid + p * blockDim.x;
        if (pixel_idx >= out_height * out_width) continue;
        
        const int out_y = pixel_idx / out_width;
        const int out_x = pixel_idx % out_width;
        
        float sum = 0.0f;
        
        // Convolution computation
        for (int ic = 0; ic < in_ch_per_group; ++ic) {
            for (int ky = 0; ky < kernel_h; ++ky) {
                for (int kx = 0; kx < kernel_w; ++kx) {
                    const int in_y = out_y * stride_h - padding_h + ky * dilation_h;
                    const int in_x = out_x * stride_w - padding_w + kx * dilation_w;
                    
                    if (in_y >= 0 && in_y < height && in_x >= 0 && in_x < width) {
                        const int input_idx = batch_idx * in_channels * height * width +
                                              (group_idx * in_ch_per_group + ic) * height * width +
                                              in_y * width + in_x;
                        const int weight_idx = ic * kernel_h * kernel_w + ky * kernel_w + kx;
                        sum += input[input_idx] * shared_weights[weight_idx];
                    }
                }
            }
        }
        
        // Add bias
        sum += bias[out_ch_idx];
        
        // Write to output
        const int output_idx = batch_idx * out_channels * out_height * out_width +
                               out_ch_idx * out_height * out_width +
                               out_y * out_width + out_x;
        output[output_idx] = sum;
    }
}

void conv2d_forward(
    const at::Tensor input,
    const at::Tensor weight,
    const at::Tensor bias,
    at::Tensor output,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w,
    int groups
) {
    const auto batch_size = input.size(0);
    const auto in_channels = input.size(1);
    const auto height = input.size(2);
    const auto width = input.size(3);
    
    const auto out_channels = weight.size(0);
    const auto kernel_h = weight.size(2);
    const auto kernel_w = weight.size(3);
    
    const auto out_height = (height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    const auto out_width = (width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
    
    // Set up dimensions
    const int threads_x = 32;
    const int threads_y = 8;
    const dim3 threads(threads_x, threads_y);
    
    const int blocks_x = (out_width * out_height + threads_x - 1) / threads_x;
    const int blocks_y = (out_channels + threads_y - 1) / threads_y;
    const dim3 blocks(blocks_x, blocks_y, batch_size);
    
    // Shared memory size for weights
    const int in_ch_per_group = in_channels / groups;
    const int shared_mem_size = in_ch_per_group * kernel_h * kernel_w * sizeof(float);
    
    // Launch kernel
    at::cuda::CUDAGuard device_guard(input.device());
    conv2d_kernel<<<blocks, threads, shared_mem_size>>>(
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
        dilation_w,
        groups,
        out_height,
        out_width
    );
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("CUDA kernel launch failed: ", cudaGetErrorString(err));
    }
}
"""

# C++ source code for PyBind11 bindings
cpp_source = r"""
#include <torch/extension.h>

void conv2d_forward(
    const at::Tensor input,
    const at::Tensor weight,
    const at::Tensor bias,
    at::Tensor output,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w,
    int groups
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv2d_forward", &conv2d_forward, "Conv2D forward pass");
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
    # Handle stride, padding, and dilation parameters
    if isinstance(conv1d_stride, int):
        stride_h, stride_w = conv1d_stride, conv1d_stride
    else:
        stride_h, stride_w = conv1d_stride[0], conv1d_stride[1]
        
    if isinstance(conv1d_padding, int):
        padding_h, padding_w = conv1d_padding, conv1d_padding
    else:
        padding_h, padding_w = conv1d_padding[0], conv1d_padding[1]
        
    if isinstance(conv1d_dilation, int):
        dilation_h, dilation_w = conv1d_dilation, conv1d_dilation
    else:
        dilation_h, dilation_w = conv1d_dilation[0], conv1d_dilation[1]
    
    # Create output tensor
    out_channels = conv1d_weight.size(0)
    in_height, in_width = x.size(2), x.size(3)
    out_height = (in_height + 2 * padding_h - dilation_h * (conv1d_weight.size(2) - 1) - 1) // stride_h + 1
    out_width = (in_width + 2 * padding_w - dilation_w * (conv1d_weight.size(3) - 1) - 1) // stride_w + 1
    output = torch.empty(x.size(0), out_channels, out_height, out_width, device=x.device, dtype=x.dtype)
    
    # Call the custom CUDA kernel
    conv_ext.conv2d_forward(
        x, conv1d_weight, conv1d_bias, output,
        stride_h, stride_w,
        padding_h, padding_w,
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
