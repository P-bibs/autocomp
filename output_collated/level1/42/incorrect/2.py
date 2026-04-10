# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_052448/code_8.py
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

# --- CUDA Kernel for Vectorized Max Pooling ---
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define WARP_SIZE 32

__global__ void vectorized_maxpool2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int channels,
    int input_height,
    int input_width,
    int output_height,
    int output_width,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    // Calculate global thread indices
    int batch_idx = blockIdx.x;
    int channel_idx = blockIdx.y;
    int out_y = blockIdx.z * blockDim.y + threadIdx.y;
    int out_x_start = (blockIdx.z * blockDim.x + threadIdx.x) * 4; // Process 4 elements per thread

    if (batch_idx >= batch_size || channel_idx >= channels || out_y >= output_height) return;

    // Shared memory for intermediate results (if needed for more complex optimizations)
    // extern __shared__ float shared_mem[];

    // Calculate base input index for the current batch and channel
    const float* input_base = input + (batch_idx * channels + channel_idx) * input_height * input_width;
    float* output_base = output + (batch_idx * channels + channel_idx) * output_height * output_width;

    // Vectorized processing: handle 4 consecutive output pixels
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int out_x = out_x_start + i;
        if (out_x >= output_width) break;

        // Compute input window boundaries
        int in_y_start = out_y * stride - padding;
        int in_x_start = out_x * stride - padding;

        float max_val = -INFINITY;

        // Iterate over the pooling window
        for (int ky = 0; ky < kernel_size; ++ky) {
            int in_y = in_y_start + ky * dilation;
            if (in_y < 0 || in_y >= input_height) continue;

            const float* input_row = input_base + in_y * input_width;

            for (int kx = 0; kx < kernel_size; ++kx) {
                int in_x = in_x_start + kx * dilation;
                if (in_x >= 0 && in_x < input_width) {
                    float val = input_row[in_x];
                    max_val = fmaxf(max_val, val);
                }
            }
        }

        output_base[out_y * output_width + out_x] = max_val;
    }
}

void launch_vectorized_maxpool2d(
    const torch::Tensor& input,
    torch::Tensor& output,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    int batch_size = input.size(0);
    int channels = input.size(1);
    int input_height = input.size(2);
    int input_width = input.size(3);
    int output_height = output.size(2);
    int output_width = output.size(3);

    // Grid and block dimensions
    dim3 grid(batch_size, channels, (output_height + 7) / 8); // 8 rows per block
    dim3 block(32, 8); // 32 threads per row, 8 rows per block

    vectorized_maxpool2d_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        input_height,
        input_width,
        output_height,
        output_width,
        kernel_size,
        stride,
        padding,
        dilation
    );

    cudaDeviceSynchronize(); // Ensure completion (optional based on use case)
}
"""

# --- C++ Bindings ---
cpp_source = r"""
#include <torch/extension.h>

void launch_vectorized_maxpool2d(
    const torch::Tensor& input,
    torch::Tensor& output,
    int kernel_size,
    int stride,
    int padding,
    int dilation
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vectorized_maxpool2d", &launch_vectorized_maxpool2d, "Vectorized MaxPool2D forward pass");
}
"""

# Compile the extension
vectorized_maxpool_ext = load_inline(
    name='vectorized_maxpool',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math', '-lineinfo'],
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
    # Handle unsupported features in our custom kernel
    if maxpool_ceil_mode or maxpool_return_indices:
        # Fall back to PyTorch's implementation for these rare cases
        return F.max_pool2d(x, kernel_size=maxpool_kernel_size, stride=maxpool_stride, padding=maxpool_padding, dilation=maxpool_dilation, ceil_mode=maxpool_ceil_mode, return_indices=maxpool_return_indices)
    
    # Calculate output dimensions
    out_h = ((x.shape[2] + 2 * maxpool_padding - maxpool_dilation * (maxpool_kernel_size - 1) - 1) // maxpool_stride) + 1
    out_w = ((x.shape[3] + 2 * maxpool_padding - maxpool_dilation * (maxpool_kernel_size - 1) - 1) // maxpool_stride) + 1
    
    # Create output tensor
    output = torch.empty(x.shape[0], x.shape[1], out_h, out_w, dtype=x.dtype, device=x.device)
    
    # Launch custom CUDA kernel
    vectorized_maxpool_ext.vectorized_maxpool2d(
        x, output,
        maxpool_kernel_size,
        maxpool_stride,
        maxpool_padding,
        maxpool_dilation
    )
    
    return output

# Test parameters
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
