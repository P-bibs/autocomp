# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_052025/code_0.py
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
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void max_pool2d_kernel(
    const float* input,
    float* output,
    int* indices,
    int batch_size,
    int channels,
    int input_height,
    int input_width,
    int output_height,
    int output_width,
    int kernel_size,
    int stride,
    int padding,
    bool return_indices) {
    
    int batch_idx = blockIdx.x;
    int channel_idx = blockIdx.y;
    int output_row = blockIdx.z * blockDim.y + threadIdx.y;
    int output_col = threadIdx.x;
    
    if (batch_idx >= batch_size || channel_idx >= channels || 
        output_row >= output_height || output_col >= output_width) {
        return;
    }
    
    int input_base_idx = ((batch_idx * channels + channel_idx) * input_height * input_width);
    int output_base_idx = ((batch_idx * channels + channel_idx) * output_height * output_width);
    
    int input_row_start = output_row * stride - padding;
    int input_col_start = output_col * stride - padding;
    int input_row_end = input_row_start + kernel_size;
    int input_col_end = input_col_start + kernel_size;
    
    // Clamp to valid input bounds
    input_row_start = max(input_row_start, 0);
    input_col_start = max(input_col_start, 0);
    input_row_end = min(input_row_end, input_height);
    input_col_end = min(input_col_end, input_width);
    
    float max_val = -INFINITY;
    int max_idx = -1;
    
    for (int i = input_row_start; i < input_row_end; i++) {
        for (int j = input_col_start; j < input_col_end; j++) {
            int input_idx = input_base_idx + i * input_width + j;
            float val = input[input_idx];
            if (val > max_val) {
                max_val = val;
                max_idx = input_idx;
            }
        }
    }
    
    int output_idx = output_base_idx + output_row * output_width + output_col;
    output[output_idx] = max_val;
    
    if (return_indices) {
        indices[output_idx] = max_idx;
    }
}

void max_pool2d_forward(
    const torch::Tensor& input,
    torch::Tensor& output,
    torch::Tensor& indices,
    int kernel_size,
    int stride,
    int padding,
    bool return_indices) {
    
    int batch_size = input.size(0);
    int channels = input.size(1);
    int input_height = input.size(2);
    int input_width = input.size(3);
    int output_height = output.size(2);
    int output_width = output.size(3);
    
    dim3 grid(batch_size, channels, (output_height + 7) / 8);
    dim3 block(min(output_width, 512), min(8, 512 / min(output_width, 512)));
    
    max_pool2d_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        indices.data_ptr<int>(),
        batch_size,
        channels,
        input_height,
        input_width,
        output_height,
        output_width,
        kernel_size,
        stride,
        padding,
        return_indices
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void max_pool2d_forward(
    const torch::Tensor& input,
    torch::Tensor& output,
    torch::Tensor& indices,
    int kernel_size,
    int stride,
    int padding,
    bool return_indices);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("max_pool2d_forward", &max_pool2d_forward, "Max Pooling 2D forward");
}
"""

fused_ext = load_inline(
    name='max_pool2d_cuda',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(
    x,
    *,
    maxpool_kernel_size,
    maxpool_stride,
    maxpool_padding,
    maxpool_dilation,
    maxpool_ceil_mode,
    maxpool_return_indices,
):
    # Handle dilation (our kernel assumes dilation=1)
    if maxpool_dilation != 1:
        raise NotImplementedError("Dilation != 1 is not supported in this optimized version")
    
    # Calculate output dimensions
    if maxpool_ceil_mode:
        output_height = int(torch.ceil(torch.tensor((x.shape[2] + 2 * maxpool_padding - maxpool_dilation * (maxpool_kernel_size - 1) - 1) / maxpool_stride + 1)).item())
        output_width = int(torch.ceil(torch.tensor((x.shape[3] + 2 * maxpool_padding - maxpool_dilation * (maxpool_kernel_size - 1) - 1) / maxpool_stride + 1)).item())
    else:
        output_height = int(torch.floor(torch.tensor((x.shape[2] + 2 * maxpool_padding - maxpool_dilation * (maxpool_kernel_size - 1) - 1) / maxpool_stride + 1)).item())
        output_width = int(torch.floor(torch.tensor((x.shape[3] + 2 * maxpool_padding - maxpool_dilation * (maxpool_kernel_size - 1) - 1) / maxpool_stride + 1)).item())
    
    # Create output tensor
    output = torch.empty((x.shape[0], x.shape[1], output_height, output_width), device=x.device, dtype=x.dtype)
    indices = torch.empty((x.shape[0], x.shape[1], output_height, output_width), device=x.device, dtype=torch.int32)
    
    # Call custom CUDA kernel
    fused_ext.max_pool2d_forward(x, output, indices, maxpool_kernel_size, maxpool_stride, maxpool_padding, maxpool_return_indices)
    
    if maxpool_return_indices:
        # Convert indices to the expected format (batch, channel, height, width)
        return output, indices
    else:
        return output

batch_size = 32
channels = 64
height = 512
width = 512
kernel_size = 4
stride = 1
padding = 1
dilation = 1

def get_init_inputs():
    return [kernel_size, stride, padding, dilation]

def get_inputs():
    x = torch.rand(batch_size, channels, height, width, device='cuda')
    return [x]
