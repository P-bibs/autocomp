# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_114736/code_0.py
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

# CUDA kernel for fused average pooling operation
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void avg_pool1d_kernel(
    const float* input,
    float* output,
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
    
    // Calculate sum
    float sum = 0.0f;
    int count = 0;
    
    if (count_include_pad) {
        // Include padded values in the count
        for (int i = input_start; i < input_end; ++i) {
            if (i >= 0 && i < input_length) {
                int input_idx = ((batch * channels + channel) * input_length) + i;
                sum += input[input_idx];
            }
        }
        // Normalize by total kernel size (include padded areas)
        if (kernel_size > 0) {
            sum /= static_cast<float>(kernel_size);
        }
    } else {
        // Only count actual input values
        count = clamped_end - clamped_start;
        if (count > 0) {
            for (int i = clamped_start; i < clamped_end; ++i) {
                int input_idx = ((batch * channels + channel) * input_length) + i;
                sum += input[input_idx];
            }
            // Normalize by actual count
            sum /= static_cast<float>(count);
        }
    }
    
    // Store result
    output[idx] = sum;
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
    // Get tensor dimensions
    auto input_sizes = input.sizes();
    int batch_size = input_sizes[0];
    int channels = input_sizes[1];
    int input_length = input_sizes[2];
    
    auto output_sizes = output.sizes();
    int output_length = output_sizes[2];
    
    // Calculate grid and block dimensions
    int total_elements = batch_size * channels * output_length;
    int threads_per_block = 256;  // Multiple of 32 for better occupancy
    int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
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
        AT_ERROR("CUDA kernel launch failed: ", cudaGetErrorString(err));
    }
}
"""

# C++ source for PyBind11 bindings
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
    m.def("avg_pool1d_cuda", &avg_pool1d_forward, "1D Average Pooling CUDA implementation");
}
"""

# Compile the extension
custom_avg_pool = load_inline(
    name='custom_avg_pool',
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
        output_length = int(torch.ceil(torch.tensor((x.size(2) + 2 * avg_pool_padding - avg_pool_kernel_size) / avg_pool_stride + 1)).item())
    else:
        output_length = int(torch.floor(torch.tensor((x.size(2) + 2 * avg_pool_padding - avg_pool_kernel_size) / avg_pool_stride + 1)).item())
    
    # Create output tensor
    output = torch.empty((x.size(0), x.size(1), output_length), device=x.device, dtype=x.dtype)
    
    # Call custom CUDA kernel
    custom_avg_pool.avg_pool1d_cuda(
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
    x = torch.rand(batch_size, in_channels, input_length, device='cuda')
    return [x]
