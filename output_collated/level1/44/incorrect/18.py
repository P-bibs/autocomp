# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_114736/code_4.py
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

# CUDA kernel for 1D average pooling
# We optimize by using coalesced memory access where each thread computes 
# one output element. For deeper optimization, shared memory tiling could 
# be used, but for kernel_size=8, this is already highly efficient.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void avg_pool1d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int channels,
    int input_length,
    int output_length,
    int kernel_size,
    int stride,
    int padding,
    bool count_include_pad
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * output_length;
    
    if (idx >= total_elements) return;
    
    int out_pos = idx % output_length;
    int channel_total = idx / output_length;
    int channel = channel_total % channels;
    int batch = channel_total / channels;
    
    int input_start = out_pos * stride - padding;
    int input_end = input_start + kernel_size;
    
    int clamped_start = max(0, input_start);
    int clamped_end = min(input_length, input_end);
    
    float sum = 0.0f;
    int count = count_include_pad ? kernel_size : (clamped_end - clamped_start);
    
    // Base pointer for this (batch, channel)
    const float* input_ptr = input + (batch * channels + channel) * input_length;
    
    if (count > 0) {
        for (int i = clamped_start; i < clamped_end; ++i) {
            sum += input_ptr[i];
        }
        output[idx] = sum / static_cast<float>(count);
    } else {
        output[idx] = 0.0f;
    }
}

void avg_pool1d_forward(
    const torch::Tensor& input,
    torch::Tensor& output,
    int kernel_size,
    int stride,
    int padding,
    bool count_include_pad
) {
    auto sizes = input.sizes();
    int batch_size = sizes[0];
    int channels = sizes[1];
    int output_length = output.sizes()[2];
    int input_length = sizes[2];
    
    int total_elements = batch_size * channels * output_length;
    int threads_per_block = 256;
    int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    avg_pool1d_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, channels, input_length, output_length,
        kernel_size, stride, padding, count_include_pad
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void avg_pool1d_forward(
    const torch::Tensor& input,
    torch::Tensor& output,
    int kernel_size,
    int stride,
    int padding,
    bool count_include_pad
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("avg_pool1d_cuda", &avg_pool1d_forward, "1D Average Pooling CUDA");
}
"""

# Compile the extension
custom_avg_pool = load_inline(
    name='custom_avg_pool',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3'],
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
    input_length = x.size(2)
    if avg_pool_ceil_mode:
        output_length = (input_length + 2 * avg_pool_padding - avg_pool_kernel_size + avg_pool_stride - 1) // avg_pool_stride + 1
    else:
        output_length = (input_length + 2 * avg_pool_padding - avg_pool_kernel_size) // avg_pool_stride + 1
    
    output = torch.empty((x.size(0), x.size(1), output_length), device=x.device, dtype=x.dtype)
    
    custom_avg_pool.avg_pool1d_cuda(
        x, output, 
        avg_pool_kernel_size, 
        avg_pool_stride, 
        avg_pool_padding, 
        avg_pool_count_include_pad
    )
    
    return output
