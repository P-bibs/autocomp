# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_052448/code_11.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['kernel_size', 'stride', 'padding', 'dilation']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['maxpool_kernel_size', 'maxpool_stride', 'maxpool_padding', 'maxpool_dilation', 'maxpool_ceil_mode', 'maxpool_return_indices']
REQUIRED_FLAT_STATE_NAMES = []


class ModelNew(nn.Module):
    """
    Simple model that performs Max Pooling 2D.
    """

    def __init__(self, kernel_size: int, stride: int, padding: int, dilation: int):
        """
        Initializes the Max Pooling 2D layer.

        Args:
            kernel_size (int): Size of the pooling window.
            stride (int): Stride of the pooling window.
            padding (int): Padding to be applied before pooling.
            dilation (int): Spacing between kernel elements.
        """
        super(ModelNew, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)

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
    # State for maxpool (nn.MaxPool2d)
    state_kwargs['maxpool_kernel_size'] = model.maxpool.kernel_size
    state_kwargs['maxpool_stride'] = model.maxpool.stride
    state_kwargs['maxpool_padding'] = model.maxpool.padding
    state_kwargs['maxpool_dilation'] = model.maxpool.dilation
    state_kwargs['maxpool_ceil_mode'] = model.maxpool.ceil_mode
    state_kwargs['maxpool_return_indices'] = model.maxpool.return_indices
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

# Define the CUDA kernel and C++ interface
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void max_pool2d_kernel(
    const float* __restrict__ input,
    float* output,
    int batch_size,
    int channels,
    int height,
    int width,
    int out_height,
    int out_width,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    int b = blockIdx.x / channels;
    int c = blockIdx.x % channels;
    int idx = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (idx < out_height * out_width) {
        int out_h = idx / out_width;
        int out_w = idx % out_width;
        
        int input_h_start = out_h * stride - padding;
        int input_w_start = out_w * stride - padding;
        
        float max_val = -INFINITY;
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int h = input_h_start + kh * dilation;
                int w = input_w_start + kw * dilation;
                if (h >= 0 && h < height && w >= 0 && w < width) {
                    int input_idx = ((b * channels + c) * height + h) * width + w;
                    float val = input[input_idx];
                    max_val = fmaxf(max_val, val);
                }
            }
        }
        
        int output_idx = ((b * channels + c) * out_height + out_h) * out_width + out_w;
        output[output_idx] = max_val;
    }
}

void max_pool2d_cuda(
    at::Tensor input,
    at::Tensor output,
    int batch_size,
    int channels,
    int height,
    int width,
    int out_height,
    int out_width,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    const float* input_data = input.data_ptr<float>();
    float* output_data = output.data_ptr<float>();
    
    int threads = 256;
    int blocks_per_batch_channel = (out_height * out_width + threads - 1) / threads;
    
    dim3 grid(batch_size * channels, blocks_per_batch_channel);
    dim3 block(threads);
    
    max_pool2d_kernel<<<grid, block>>>(
        input_data,
        output_data,
        batch_size,
        channels,
        height,
        width,
        out_height,
        out_width,
        kernel_size,
        stride,
        padding,
        dilation
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void max_pool2d_cuda(
    at::Tensor input,
    at::Tensor output,
    int batch_size,
    int channels,
    int height,
    int width,
    int out_height,
    int out_width,
    int kernel_size,
    int stride,
    int padding,
    int dilation
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("max_pool2d_cuda", &max_pool2d_cuda, "Max pooling 2D CUDA kernel");
}
"""

# Compile the extension
max_pool_ext = load_inline(
    name='max_pool2d_cuda',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# Parameters (fixed as in original code)
batch_size = 32
channels = 64
height = 512
width = 512
kernel_size = 4
stride = 1
padding = 1
dilation = 1

# Compute output dimensions
out_height = (height + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
out_width = (width + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

def get_init_inputs():
    return [kernel_size, stride, padding, dilation]

def get_inputs():
    x = torch.rand(batch_size, channels, height, width)
    return [x]

def functional_model(
    x,
    *,
    maxpool_kernel_size,
    maxpool_stride,
    maxpool_padding,
    maxpool_dilation,
    maxpool_ceil_mode,
    maxpool_return_indices,
):
    # Create output tensor on the same device as input
    output = torch.empty((batch_size, channels, out_height, out_width), dtype=x.dtype, device=x.device)
    
    # Call the custom CUDA kernel
    max_pool_ext.max_pool2d_cuda(
        x,
        output,
        batch_size,
        channels,
        height,
        width,
        out_height,
        out_width,
        maxpool_kernel_size,
        maxpool_stride,
        maxpool_padding,
        maxpool_dilation
    )
    
    return output
