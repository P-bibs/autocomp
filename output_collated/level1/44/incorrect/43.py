# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_120739/code_0.py
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
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused average pooling
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

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
    bool ceil_mode,
    bool count_include_pad
) {
    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Total number of output elements
    int total_output_elements = batch_size * channels * output_length;
    
    if (idx >= total_output_elements) return;
    
    // Decompose linear index into 3D coordinates
    int out_pos = idx % output_length;
    int temp = idx / output_length;
    int channel = temp % channels;
    int batch = temp / channels;
    
    // Calculate input starting position
    int input_start = out_pos * stride - padding;
    int input_end = input_start + kernel_size;
    
    // Clamp to valid input range
    int clamped_start = max(0, input_start);
    int clamped_end = min(input_length, input_end);
    
    float sum = 0.0f;
    int count = 0;
    
    // Sum over the kernel window
    for (int i = clamped_start; i < clamped_end; ++i) {
        int input_idx = ((batch * channels + channel) * input_length) + i;
        sum += input[input_idx];
        count++;
    }
    
    // Handle padding
    if (!count_include_pad && padding > 0) {
        // In this mode, we only count actual input elements, which we already do
        // The count is correctly calculated above
    }
    
    // Calculate denominator
    float denom = count_include_pad ? static_cast<float>(kernel_size) : static_cast<float>(count);
    
    // Avoid division by zero
    if (denom > 0) {
        output[idx] = sum / denom;
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
    bool ceil_mode,
    bool count_include_pad
) {
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto input_length = input.size(2);
    
    auto output_length = output.size(2);
    
    // Calculate total number of output elements
    int total_output_elements = batch_size * channels * output_length;
    
    // Configure kernel launch parameters
    const int threads_per_block = 256;
    const int blocks = (total_output_elements + threads_per_block - 1) / threads_per_block;
    
    // Launch kernel
    avg_pool1d_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        input_length,
        output_length,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad
    );
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR(cudaGetErrorString(err));
    }
}
"""

# C++ interface bindings
cpp_source = r"""
#include <torch/extension.h>

void avg_pool1d_forward(
    const torch::Tensor& input,
    torch::Tensor& output,
    int kernel_size,
    int stride,
    int padding,
    bool ceil_mode,
    bool count_include_pad
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("avg_pool1d_forward", &avg_pool1d_forward, "Average Pooling 1D forward pass");
}
"""

# Compile the custom extension
custom_pool_ext = load_inline(
    name='custom_avg_pool1d',
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
    # Calculate output size
    if avg_pool_ceil_mode:
        output_length = int(torch.ceil(torch.tensor((x.shape[2] + 2 * avg_pool_padding - avg_pool_kernel_size) / avg_pool_stride + 1)).item())
    else:
        output_length = int(torch.floor(torch.tensor((x.shape[2] + 2 * avg_pool_padding - avg_pool_kernel_size) / avg_pool_stride + 1)).item())
    
    # Create output tensor
    output = torch.empty(x.shape[0], x.shape[1], output_length, dtype=x.dtype, device=x.device)
    
    # Call custom CUDA kernel
    custom_pool_ext.avg_pool1d_forward(
        x, output, 
        avg_pool_kernel_size, 
        avg_pool_stride, 
        avg_pool_padding, 
        avg_pool_ceil_mode, 
        avg_pool_count_include_pad
    )
    
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
    x = torch.rand(batch_size, in_channels, input_length)
    return [x]
