# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_052448/code_23.py
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
from torch.utils.cpp_extension import load_inline

# CUDA kernel optimized for parallel output generation
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

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
    // Each thread calculates one output element: (batch, channel, oh, ow)
    int linear_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * out_height * out_width;

    if (linear_idx < total_elements) {
        int ow = linear_idx % out_width;
        int temp = linear_idx / out_width;
        int oh = temp % out_height;
        int c = (temp / out_height) % channels;
        int b = (temp / out_height) / channels;

        int iy_start = oh * stride - padding;
        int ix_start = ow * stride - padding;

        float max_val = -1e38f; // Smaller than any reasonable float32

        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int iy = iy_start + kh * dilation;
                int ix = ix_start + kw * dilation;

                if (iy >= 0 && iy < height && ix >= 0 && ix < width) {
                    float val = input[((b * channels + c) * height + iy) * width + ix];
                    if (val > max_val) max_val = val;
                }
            }
        }
        output[linear_idx] = max_val;
    }
}

void max_pool2d_cuda_launch(
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
    const int total_elements = batch_size * channels * out_height * out_width;
    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;
    
    max_pool2d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
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
void max_pool2d_cuda_launch(at::Tensor input, at::Tensor output, int batch_size, int channels, int height, int width, int out_height, int out_width, int kernel_size, int stride, int padding, int dilation);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_cuda_launch, "Max Pool 2D Forward");
}
"""

module = load_inline(
    name='max_pool_kernel',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, maxpool_kernel_size, maxpool_stride, maxpool_padding, maxpool_dilation, maxpool_ceil_mode=False, maxpool_return_indices=False):
    # Calculate output dims
    b, c, h, w = x.shape
    oh = (h + 2 * maxpool_padding - maxpool_dilation * (maxpool_kernel_size - 1) - 1) // maxpool_stride + 1
    ow = (w + 2 * maxpool_padding - maxpool_dilation * (maxpool_kernel_size - 1) - 1) // maxpool_stride + 1
    
    output = torch.empty((b, c, oh, ow), device=x.device, dtype=x.dtype)
    
    module.forward(x, output, b, c, h, w, oh, ow, maxpool_kernel_size, maxpool_stride, maxpool_padding, maxpool_dilation)
    
    return output
