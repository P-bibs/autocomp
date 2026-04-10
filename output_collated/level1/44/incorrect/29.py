# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_115947/code_0.py
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

# --- CUDA Kernel ---
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void avg_pool1d_fused_kernel(
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
    int total_output_elements = batch_size * channels * output_length;
    
    if (idx >= total_output_elements) return;
    
    // Decompose linear index into 3D coordinates
    int out_pos = idx % output_length;
    int temp = idx / output_length;
    int ch = temp % channels;
    int batch = temp / channels;
    
    // Calculate input position
    int in_start = out_pos * stride - padding;
    int in_end = in_start + kernel_size;
    
    // Clamp to valid input range
    in_start = max(in_start, 0);
    in_end = min(in_end, input_length);
    
    // Compute average
    float sum = 0.0f;
    int count = 0;
    
    if (count_include_pad) {
        // Count all positions in the window, including padded ones
        count = in_end - in_start;
        for (int i = in_start; i < in_end; ++i) {
            int input_idx = ((batch * channels + ch) * input_length) + i;
            sum += input[input_idx];
        }
    } else {
        // Only count actual input elements (exclude padded regions)
        for (int i = in_start; i < in_end; ++i) {
            int input_idx = ((batch * channels + ch) * input_length) + i;
            sum += input[input_idx];
            count++;
        }
    }
    
    // Write result
    if (count > 0) {
        output[idx] = sum / static_cast<float>(count);
    } else {
        output[idx] = 0.0f;
    }
}

void avg_pool1d_fused_launcher(
    const torch::Tensor& input,
    torch::Tensor& output,
    int kernel_size,
    int stride,
    int padding,
    bool ceil_mode,
    bool count_include_pad
) {
    auto input_sizes = input.sizes();
    int batch_size = input_sizes[0];
    int channels = input_sizes[1];
    int input_length = input_sizes[2];
    
    auto output_sizes = output.sizes();
    int output_length = output_sizes[2];
    
    const int threads_per_block = 256;
    const int total_elements = batch_size * channels * output_length;
    const int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    avg_pool1d_fused_kernel<<<blocks, threads_per_block>>>(
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
}
"""

# --- C++ Logic (Interface/Bindings) ---
cpp_source = r"""
#include <torch/extension.h>

void avg_pool1d_fused_launcher(
    const torch::Tensor& input,
    torch::Tensor& output,
    int kernel_size,
    int stride,
    int padding,
    bool ceil_mode,
    bool count_include_pad
);

void avg_pool1d_fused(
    const torch::Tensor& input,
    torch::Tensor& output,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    bool ceil_mode,
    bool count_include_pad
) {
    avg_pool1d_fused_launcher(input, output, static_cast<int>(kernel_size), 
                              static_cast<int>(stride), static_cast<int>(padding),
                              ceil_mode, count_include_pad);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("avg_pool1d_fused", &avg_pool1d_fused, "Fused AvgPool1D forward pass");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='avg_pool1d_fused_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# Global variables
batch_size = 64
in_channels = 128
input_length = 65536
avg_pool_kernel_size = 8
avg_pool_stride = 1
avg_pool_padding = 4
avg_pool_ceil_mode = False
avg_pool_count_include_pad = True

# Precompute output length (matching PyTorch's behavior)
def compute_output_length(input_length, kernel_size, stride, padding, ceil_mode):
    if ceil_mode:
        return int(torch.ceil(torch.tensor((input_length + 2 * padding - kernel_size) / stride + 1)).item())
    else:
        return (input_length + 2 * padding - kernel_size) // stride + 1

output_length = compute_output_length(
    input_length, 
    avg_pool_kernel_size, 
    avg_pool_stride, 
    avg_pool_padding, 
    avg_pool_ceil_mode
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
    # Create output tensor with correct shape
    output_shape = (x.size(0), x.size(1), output_length)
    output = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    
    # Call fused CUDA kernel
    fused_ext.avg_pool1d_fused(
        x, 
        output, 
        avg_pool_kernel_size, 
        avg_pool_stride, 
        avg_pool_padding,
        avg_pool_ceil_mode,
        avg_pool_count_include_pad
    )
    return output

def get_init_inputs():
    return [avg_pool_kernel_size, avg_pool_stride, avg_pool_padding]

def get_inputs():
    x = torch.rand(batch_size, in_channels, input_length, device='cuda')
    return [x]
