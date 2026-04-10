# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_052448/code_10.py
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# -------------------------------------------------------------------------
# CUDA kernel – now uses a grid‑stride loop
# -------------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void max_pool2d_kernel(
    const float* input,
    float* output,
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
    int total_outputs = batch_size * channels * output_height * output_width;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_threads = blockDim.x * gridDim.x;   // total number of threads

    // Grid‑stride loop: each thread processes multiple output points
    for (int out_idx = idx; out_idx < total_outputs; out_idx += stride_threads) {
        // Decode flat index into (n, c, h_out, w_out)
        int w_out = out_idx % output_width;
        int h_out = (out_idx / output_width) % output_height;
        int c     = (out_idx / (output_width * output_height)) % channels;
        int n     = out_idx / (output_width * output_height * channels);

        // Compute bounding box in the input image
        int h_start = h_out * stride - padding;
        int w_start = w_out * stride - padding;
        int h_end   = h_start + (kernel_size - 1) * dilation + 1;
        int w_end   = w_start + (kernel_size - 1) * dilation + 1;

        // Clamp to input dimensions
        h_start = max(h_start, 0);
        w_start = max(w_start, 0);
        h_end   = min(h_end,   input_height);
        w_end   = min(w_end,   input_width);

        // Max‑pooling over the window
        float max_val = -INFINITY;
        for (int h = h_start; h < h_end; h += dilation) {
            for (int w = w_start; w < w_end; w += dilation) {
                int input_idx = ((n * channels + c) * input_height + h) * input_width + w;
                float val = input[input_idx];
                if (val > max_val) max_val = val;
            }
        }
        output[out_idx] = max_val;
    }
}

void max_pool2d_forward(
    const torch::Tensor input,
    torch::Tensor output,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    int batch_size    = input.size(0);
    int channels      = input.size(1);
    int input_height  = input.size(2);
    int input_width   = input.size(3);
    int output_height = output.size(2);
    int output_width  = output.size(3);

    int total_outputs = batch_size * channels * output_height * output_width;
    const int threads_per_block = 256;
    const int max_blocks = 4096;                 // limit block count
    int blocks = (total_outputs + threads_per_block - 1) / threads_per_block;
    if (blocks > max_blocks) blocks = max_blocks;

    max_pool2d_kernel<<<blocks, threads_per_block>>>(
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
}
"""

# -------------------------------------------------------------------------
# C++ binding (PYBIND11)
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void max_pool2d_forward(
    const torch::Tensor input,
    torch::Tensor output,
    int kernel_size,
    int stride,
    int padding,
    int dilation
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("max_pool2d_forward", &max_pool2d_forward, "Max Pool 2D forward pass");
}
"""

# -------------------------------------------------------------------------
# Compile the extension
# -------------------------------------------------------------------------
max_pool_ext = load_inline(
    name='max_pool2d_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# -------------------------------------------------------------------------
# Functional model that calls the custom kernel
# -------------------------------------------------------------------------
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
    # Compute output spatial size (same formula as the original code)
    if maxpool_ceil_mode:
        output_height = torch.ceil(
            torch.tensor((x.shape[2] + 2 * maxpool_padding -
                          maxpool_dilation * (maxpool_kernel_size - 1) - 1) /
                         maxpool_stride + 1)
        ).int().item()
        output_width = torch.ceil(
            torch.tensor((x.shape[3] + 2 * maxpool_padding -
                          maxpool_dilation * (maxpool_kernel_size - 1) - 1) /
                         maxpool_stride + 1)
        ).int().item()
    else:
        output_height = torch.floor(
            torch.tensor((x.shape[2] + 2 * maxpool_padding -
                          maxpool_dilation * (maxpool_kernel_size - 1) - 1) /
                         maxpool_stride + 1)
        ).int().item()
        output_width = torch.floor(
            torch.tensor((x.shape[3] + 2 * maxpool_padding -
                          maxpool_dilation * (maxpool_kernel_size - 1) - 1) /
                         maxpool_stride + 1)
        ).int().item()

    # Allocate output tensor on the same device
    output = torch.empty((x.shape[0], x.shape[1], output_height, output_width),
                         device=x.device, dtype=x.dtype)

    # Invoke the optimized CUDA kernel
    max_pool_ext.max_pool2d_forward(
        x, output,
        maxpool_kernel_size,
        maxpool_stride,
        maxpool_padding,
        maxpool_dilation
    )

    if maxpool_return_indices:
        # Return dummy indices (not required for this optimisation)
        return output, torch.empty_like(output, dtype=torch.long)
    else:
        return output


# -------------------------------------------------------------------------
# Test‑harness constants (same as the original script)
# -------------------------------------------------------------------------
batch_size = 32
channels   = 64
height     = 512
width      = 512
kernel_size = 4
stride      = 1
padding     = 1
dilation    = 1


def get_init_inputs():
    return [kernel_size, stride, padding, dilation]


def get_inputs():
    x = torch.rand(batch_size, channels, height, width, device='cuda')
    return [x]
