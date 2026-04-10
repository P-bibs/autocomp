# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_055939/code_21.py
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

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void max_pool2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int channels,
    const int in_h, const int in_w,
    const int out_h, const int out_w,
    const int k_size, const int stride, const int padding) {

    // Map grid to (out_w, out_h, batch_size * channels)
    // To handle large batch/channel sizes, we map blockIdx.z = batch and blockIdx.y = channel
    const int b = blockIdx.z;
    const int c = blockIdx.y;
    const int ow = blockIdx.x * blockDim.x + threadIdx.x;
    const int oh = threadIdx.y; 

    if (ow < out_w && oh < out_h) {
        const int input_batch_channel_base = (b * channels + c) * in_h * in_w;
        const int output_idx = ((b * channels + c) * out_h + oh) * out_w + ow;

        float max_val = -1e38f;
        const int ih_start = oh * stride - padding;
        const int iw_start = ow * stride - padding;

        for (int i = 0; i < k_size; ++i) {
            const int ih = ih_start + i;
            if (ih >= 0 && ih < in_h) {
                const int input_row_start = input_batch_channel_base + ih * in_w;
                for (int j = 0; j < k_size; ++j) {
                    const int iw = iw_start + j;
                    if (iw >= 0 && iw < in_w) {
                        float val = input[input_row_start + iw];
                        if (val > max_val) max_val = val;
                    }
                }
            }
        }
        output[output_idx] = max_val;
    }
}

void max_pool2d_forward(const torch::Tensor& input, torch::Tensor& output,
                        int k_size, int stride, int padding) {
    const int batch = input.size(0);
    const int channels = input.size(1);
    const int out_h = output.size(2);
    const int out_w = output.size(3);

    // Block configuration: X maps to width, Y maps to height
    dim3 block(32, out_h);
    dim3 grid((out_w + 31) / 32, channels, batch);

    max_pool2d_kernel<<<grid, block>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        channels, input.size(2), input.size(3), out_h, out_w,
        k_size, stride, padding
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void max_pool2d_forward(const torch::Tensor& input, torch::Tensor& output, int k_size, int stride, int padding);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_forward, "Max pool 2D forward");
}
"""

# Compile the extension
maxpool_ext = load_inline(
    name='max_pool2d_optimized',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, maxpool_kernel_size, maxpool_stride, maxpool_padding,
                     maxpool_dilation, maxpool_ceil_mode, maxpool_return_indices):
    h_in, w_in = x.shape[2], x.shape[3]
    h_out = (h_in + 2 * maxpool_padding - maxpool_kernel_size) // maxpool_stride + 1
    w_out = (w_in + 2 * maxpool_padding - maxpool_kernel_size) // maxpool_stride + 1
    
    output = torch.empty((x.shape[0], x.shape[1], h_out, w_out), 
                         device=x.device, dtype=x.dtype)
    
    maxpool_ext.forward(x.contiguous(), output, 
                        maxpool_kernel_size, maxpool_stride, maxpool_padding)
    
    return output
