# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_113602/code_4.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['kernel_size', 'stride', 'padding']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['avg_pool_kernel_size', 'avg_pool_stride', 'avg_pool_padding', 'avg_pool_ceil_mode', 'avg_pool_count_include_pad']
REQUIRED_FLAT_STATE_NAMES = []


class ModelNew(nn.Module):
    """
    Simple model that performs 1D Average Pooling.
    """

    def __init__(self, kernel_size: int, stride: int=1, padding: int=0):
        """
        Initializes the 1D Average Pooling layer.

        Args:
            kernel_size (int): Size of the pooling window.
            stride (int, optional): Stride of the pooling operation. Defaults to 1.
            padding (int, optional): Padding applied to the input tensor. Defaults to 0.
        """
        super(ModelNew, self).__init__()
        self.avg_pool = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=padding)

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
    # State for avg_pool (nn.AvgPool1d)
    state_kwargs['avg_pool_kernel_size'] = model.avg_pool.kernel_size
    state_kwargs['avg_pool_stride'] = model.avg_pool.stride
    state_kwargs['avg_pool_padding'] = model.avg_pool.padding
    state_kwargs['avg_pool_ceil_mode'] = model.avg_pool.ceil_mode
    state_kwargs['avg_pool_count_include_pad'] = model.avg_pool.count_include_pad
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

# CUDA kernel implementation for 1D Average Pooling
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void avg_pool1d_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int channels,
    const int input_length,
    const int output_length,
    const int kernel_size,
    const int stride,
    const int padding,
    const bool count_include_pad) {
    
    // Map thread to output element
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * channels * output_length) return;

    int tmp = idx;
    int col = tmp % output_length;
    tmp /= output_length;
    int chan = tmp % channels;
    int batch = tmp / channels;

    int input_start = col * stride - padding;
    int input_end = input_start + kernel_size;
    
    int actual_start = max(input_start, 0);
    int actual_end = min(input_end, input_length);
    
    scalar_t sum = 0;
    const scalar_t* channel_ptr = input + (batch * channels + chan) * input_length;
    
    for (int i = actual_start; i < actual_end; ++i) {
        sum += channel_ptr[i];
    }
    
    int count = count_include_pad ? kernel_size : (actual_end - actual_start);
    output[idx] = count > 0 ? (sum / (scalar_t)count) : 0;
}

void avg_pool1d_forward_cuda(
    const torch::Tensor& input,
    torch::Tensor& output,
    int kernel_size, int stride, int padding, bool count_include_pad) {
    
    int batch = input.size(0);
    int channels = input.size(1);
    int out_len = output.size(2);
    int total_elements = batch * channels * out_len;
    
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "avg_pool1d_forward", ([&] {
        avg_pool1d_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch, channels, input.size(2), out_len,
            kernel_size, stride, padding, count_include_pad
        );
    }));
}
"""

cpp_source = r"""
void avg_pool1d_forward_cuda(const torch::Tensor& input, torch::Tensor& output, int kernel, int stride, int pad, bool count_pad);
"""

# Binding code
module_source = cpp_source + r"""
#include <torch/extension.h>
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("avg_pool1d", &avg_pool1d_forward_cuda, "Avg Pool 1D Forward");
}
"""

avg_pool_ext = load_inline(
    name='avg_pool_ext',
    cpp_sources=module_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(
    x,
    *,
    avg_pool_kernel_size,
    avg_pool_stride,
    avg_pool_padding,
    avg_pool_ceil_mode,
    avg_pool_count_include_pad,
):
    # Calculate output length
    if avg_pool_ceil_mode:
        output_length = (x.size(2) + 2 * avg_pool_padding - avg_pool_kernel_size + avg_pool_stride - 1) // avg_pool_stride + 1
    else:
        output_length = (x.size(2) + 2 * avg_pool_padding - avg_pool_kernel_size) // avg_pool_stride + 1
    
    output = torch.empty((x.size(0), x.size(1), output_length), dtype=x.dtype, device=x.device)
    
    avg_pool_ext.avg_pool1d(
        x, output, 
        avg_pool_kernel_size, 
        avg_pool_stride, 
        avg_pool_padding, 
        avg_pool_count_include_pad
    )
    return output
