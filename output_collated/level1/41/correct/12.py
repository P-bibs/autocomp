# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_035026/code_14.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['kernel_size', 'stride', 'padding', 'dilation', 'return_indices']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['maxpool_kernel_size', 'maxpool_stride', 'maxpool_padding', 'maxpool_dilation', 'maxpool_ceil_mode', 'maxpool_return_indices']
REQUIRED_FLAT_STATE_NAMES = []


class ModelNew(nn.Module):
    """
    Simple model that performs Max Pooling 1D.
    """

    def __init__(self, kernel_size: int, stride: int=None, padding: int=0, dilation: int=1, return_indices: bool=False):
        """
        Initializes the Max Pooling 1D layer.

        Args:
            kernel_size (int): Size of the window to take a max over.
            stride (int, optional): Stride of the window. Defaults to None (same as kernel_size).
            padding (int, optional): Implicit zero padding to be added on both sides. Defaults to 0.
            dilation (int, optional): Spacing between kernel elements. Defaults to 1.
            return_indices (bool, optional): Whether to return the indices of the maximum values. Defaults to False.
        """
        super(ModelNew, self).__init__()
        self.maxpool = nn.MaxPool1d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, return_indices=return_indices)

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
    # State for maxpool (nn.MaxPool1d)
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

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <algorithm>

__global__ void max_pool1d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int batch_size,
    const int channels,
    const int input_length,
    const int output_length,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output_elements = batch_size * channels * output_length;
    
    if (idx >= total_output_elements) return;
    
    int out_pos = idx % output_length;
    int channel_idx = (idx / output_length) % channels;
    int batch_idx = idx / (channels * output_length);
    
    int input_start = out_pos * stride - padding;
    
    float max_val = -1e38f; // Representing -inf for float
    
    // Pointer to the start of the current channel in input
    const float* channel_ptr = input + (batch_idx * channels + channel_idx) * input_length;
    
    for (int k = 0; k < kernel_size; ++k) {
        int input_pos = input_start + k * dilation;
        if (input_pos >= 0 && input_pos < input_length) {
            float val = channel_ptr[input_pos];
            if (val > max_val) {
                max_val = val;
            }
        } else if (input_pos < 0) {
            // No-op for padding area (assuming max pooling over valid values)
        }
    }
    
    output[idx] = max_val;
}

void max_pool1d_forward(
    const at::Tensor& input,
    at::Tensor& output,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    int batch_size = input.size(0);
    int channels = input.size(1);
    int input_length = input.size(2);
    int output_length = output.size(2);
    
    int total_output_elements = batch_size * channels * output_length;
    int threads = 256;
    int blocks = (total_output_elements + threads - 1) / threads;
    
    max_pool1d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        input_length,
        output_length,
        kernel_size,
        stride,
        padding,
        dilation
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void max_pool1d_forward(const at::Tensor& input, at::Tensor& output, int kernel_size, int stride, int padding, int dilation);

torch::Tensor max_pool1d_cuda(const torch::Tensor& input, int kernel_size, int stride, int padding, int dilation) {
    int input_length = input.size(2);
    int output_length = (input_length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    auto output = torch::empty({input.size(0), input.size(1), output_length}, input.options());
    max_pool1d_forward(input, output, kernel_size, stride, padding, dilation);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("max_pool1d_cuda", &max_pool1d_cuda, "optimized 1d max pool");
}
"""

fused_ext = load_inline(
    name='fused_max_pool1d',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, maxpool_kernel_size, maxpool_stride, maxpool_padding, maxpool_dilation, maxpool_ceil_mode, maxpool_return_indices):
    return fused_ext.max_pool1d_cuda(x, maxpool_kernel_size, maxpool_stride, maxpool_padding, maxpool_dilation)

batch_size = 64
features = 192
sequence_length = 65536
kernel_size = 8
stride = 1
padding = 4
dilation = 3            
return_indices = False

def get_init_inputs():
    return [kernel_size, stride, padding, dilation, return_indices]

def get_inputs():
    return [torch.rand(batch_size, features, sequence_length, device='cuda')]
