# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_120739/code_7.py
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

# ----------------------------------------------------------------------
# CUDA Kernel implementation
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void avg_pool1d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int channels_total,
    const int input_length,
    const int output_length,
    const int kernel_size,
    const int stride,
    const int padding,
    const bool count_include_pad
) {
    // Each block processes a chunk of output length for a specific batch/channel
    const int batch_chan_idx = blockIdx.y;
    const int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (out_idx >= output_length) return;

    int in_start = out_idx * stride - padding;
    float sum = 0.0f;
    int count = 0;

    for (int k = 0; k < kernel_size; ++k) {
        int in_i = in_start + k;
        if (in_i >= 0 && in_i < input_length) {
            sum += input[batch_chan_idx * input_length + in_i];
            count++;
        } else if (count_include_pad) {
            count++;
        }
    }

    float denom = (count > 0) ? static_cast<float>(count) : 1.0f;
    output[batch_chan_idx * output_length + out_idx] = sum / denom;
}

void fused_op_forward(
    at::Tensor input,
    at::Tensor output,
    int batch_size,
    int in_channels,
    int input_length,
    int output_length,
    int kernel_size,
    int stride,
    int padding,
    bool count_include_pad
) {
    const int channels_total = batch_size * in_channels;
    const int threads = 256;
    const int blocks = (output_length + threads - 1) / threads;
    
    dim3 grid(blocks, channels_total);
    
    avg_pool1d_kernel<<<grid, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        channels_total,
        input_length,
        output_length,
        kernel_size,
        stride,
        padding,
        count_include_pad
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(
    at::Tensor input,
    at::Tensor output,
    int batch_size,
    int in_channels,
    int input_length,
    int output_length,
    int kernel_size,
    int stride,
    int padding,
    bool count_include_pad
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused 1D average pooling forward");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
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
    avg_pool_count_include_pad
):
    # Ensure input is on GPU and contiguous
    x = x.cuda().contiguous()
    batch_size, in_channels, input_length = x.shape
    
    # Calculate output dimensions
    if avg_pool_ceil_mode:
        out_len = (input_length + 2 * avg_pool_padding - avg_pool_kernel_size + avg_pool_stride - 1) // avg_pool_stride + 1
    else:
        out_len = (input_length + 2 * avg_pool_padding - avg_pool_kernel_size) // avg_pool_stride + 1
    
    out = torch.empty((batch_size, in_channels, out_len), device=x.device, dtype=x.dtype)
    
    fused_ext.fused_op(
        x,
        out,
        batch_size,
        in_channels,
        input_length,
        out_len,
        avg_pool_kernel_size,
        avg_pool_stride,
        avg_pool_padding,
        avg_pool_count_include_pad
    )
    
    return out
