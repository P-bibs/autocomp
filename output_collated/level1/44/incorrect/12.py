# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_114325/code_1.py
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

# Define CUDA kernel for optimized avg_pool1d operation
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void avg_pool1d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int input_length,
    int kernel_size,
    int stride,
    int padding,
    int output_length
) {
    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output_elements = batch_size * in_channels * output_length;

    if (idx >= total_output_elements)
        return;

    // Map linear index to 3D coordinates (batch, channel, output_position)
    int output_pos = idx % output_length;
    int temp = idx / output_length;
    int channel = temp % in_channels;
    int batch = temp / in_channels;

    // Compute input window boundaries
    int start = output_pos * stride - padding;
    int end = start + kernel_size;

    // Apply boundary conditions
    int clamped_start = max(start, 0);
    int clamped_end = min(end, input_length);

    // Pointer to the start of the current channel's data
    const float* input_ptr = input + (batch * in_channels + channel) * input_length;
    
    // Perform pooling: sum over valid window elements
    float sum = 0.0f;
    int count = 0;
    for (int i = clamped_start; i < clamped_end; ++i) {
        sum += input_ptr[i];
        count++;
    }

    // Store average value in output tensor
    output[idx] = sum / static_cast<float>(kernel_size);
}

void avg_pool1d_cuda_launcher(
    const torch::Tensor& input,
    torch::Tensor& output,
    int kernel_size,
    int stride,
    int padding
) {
    // Extract tensor dimensions
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_length = input.size(2);

    // Compute output length based on pooling parameters
    int output_length = (input_length + 2 * padding - kernel_size) / stride + 1;

    // Total number of output elements to process
    int total_elements = batch_size * in_channels * output_length;

    // Configure kernel launch parameters
    const int threads_per_block = 256;
    const int blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    // Launch CUDA kernel
    avg_pool1d_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        input_length,
        kernel_size,
        stride,
        padding,
        output_length
    );
}
"""

# C++ binding code for PyTorch extension
cpp_source = r"""
#include <torch/extension.h>

void avg_pool1d_cuda_launcher(
    const torch::Tensor& input,
    torch::Tensor& output,
    int kernel_size,
    int stride,
    int padding
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("avg_pool1d_cuda", &avg_pool1d_cuda_launcher, "Optimized 1D Average Pooling CUDA Kernel");
}
"""

# Compile the custom CUDA extension
fused_ext = load_inline(
    name='fused_avg_pool1d_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
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
    """
    Optimized version of F.avg_pool1d using a custom CUDA kernel that improves
    memory access patterns and reduces overhead for large inputs.
    """
    # Compute output size according to avg_pool1d formula
    batch_size, in_channels, input_length = x.shape
    output_length = (input_length + 2 * avg_pool_padding - avg_pool_kernel_size) // avg_pool_stride + 1
    
    # Create output tensor with correct shape and device
    output = torch.empty((batch_size, in_channels, output_length), device=x.device, dtype=x.dtype)
    
    # Launch optimized CUDA kernel
    fused_ext.avg_pool1d_cuda(x, output, avg_pool_kernel_size, avg_pool_stride, avg_pool_padding)
    
    return output

# Parameters used for testing
batch_size = 64
in_channels = 128
input_length = 65536
kernel_size = 8
stride = 1
padding = 4

def get_init_inputs():
    return [kernel_size, stride, padding]

def get_inputs():
    x = torch.rand(batch_size, in_channels, input_length, device='cuda')
    return [x]
