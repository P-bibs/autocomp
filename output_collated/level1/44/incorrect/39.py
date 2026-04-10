# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_120345/code_4.py
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

# The CUDA kernel performs efficient 1D average pooling.
# It uses a grid-stride loop, which is robust for large tensors.
# The calculation uses coalesced memory access by mapping the thread index 
# (which increments contiguously across the last dimension) to the output tensor's last dimension.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void avg_pool1d_cuda_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int num_elements,
    const int channels,
    const int input_length,
    const int output_length,
    const int kernel_size,
    const int stride,
    const int padding) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    int out_pos = idx % output_length;
    int temp = idx / output_length;
    int channel = temp % channels;
    int batch = temp / channels;

    int in_start = out_pos * stride - padding;
    int in_end = in_start + kernel_size;

    int clamp_start = max(0, in_start);
    int clamp_end = min(input_length, in_end);

    float sum = 0.0f;
    int count = clamp_end - clamp_start;

    // Use pointer arithmetic to access the relevant batch and channel in input
    const float* input_ptr = input + (batch * channels + channel) * input_length;

    if (count > 0) {
        for (int j = clamp_start; j < clamp_end; j++) {
            sum += input_ptr[j];
        }
        output[idx] = sum / (float)count;
    } else {
        output[idx] = 0.0f;
    }
}

void avg_pool1d_cuda_forward(
    const torch::Tensor input,
    torch::Tensor output,
    const int kernel_size,
    const int stride,
    const int padding) {
    
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int output_length = output.size(2);
    const int num_elements = batch_size * channels * output_length;
    
    const int threads_per_block = 256;
    const int blocks = (num_elements + threads_per_block - 1) / threads_per_block;
    
    avg_pool1d_cuda_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        num_elements,
        channels,
        input.size(2),
        output_length,
        kernel_size,
        stride,
        padding
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void avg_pool1d_cuda_forward(
    const torch::Tensor input,
    torch::Tensor output,
    const int kernel_size,
    const int stride,
    const int padding);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("avg_pool1d_cuda", &avg_pool1d_cuda_forward, "Average Pooling 1D CUDA implementation");
}
"""

# Compile the extension once
avg_pool_ext = load_inline(
    name='avg_pool1d_cuda',
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
    input_length = x.shape[2]
    if avg_pool_ceil_mode:
        output_length = (input_length + 2 * avg_pool_padding - avg_pool_kernel_size + avg_pool_stride - 1) // avg_pool_stride + 1
    else:
        output_length = (input_length + 2 * avg_pool_padding - avg_pool_kernel_size) // avg_pool_stride + 1
    
    output = torch.empty(x.shape[0], x.shape[1], output_length, dtype=x.dtype, device=x.device)
    
    avg_pool_ext.avg_pool1d_cuda(x, output, avg_pool_kernel_size, avg_pool_stride, avg_pool_padding)
    
    return output

batch_size = 64
in_channels = 128
input_length = 65536
kernel_size = 8
stride = 1
padding = 4

def get_init_inputs():
    return [kernel_size, stride, padding]

def get_inputs():
    # Ensure data is on GPU as the implementation is CUDA specific
    x = torch.rand(batch_size, in_channels, input_length, device='cuda')
    return [x]
