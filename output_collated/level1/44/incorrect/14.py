# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_114325/code_7.py
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

# ------------------------------------------------------------
# 1. CUDA kernel (grid‑stride, register‑only accumulation)
# ------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void avg_pool1d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int batch_size,
    const int channels,
    const int in_len,
    const int out_len,
    const int kernel_size,
    const int stride,
    const int padding,
    const bool count_include_pad)
{
    const int total = batch_size * channels * out_len;
    // Grid‑stride loop: each thread handles multiple output positions
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += blockDim.x * gridDim.x)
    {
        // Decode linear index
        int temp = idx;
        int i_out = temp % out_len;
        temp /= out_len;
        int c = temp % channels;
        int b = temp / channels;

        int start = i_out * stride - padding;
        float sum = 0.0f;
        int count = 0;

        for (int k = 0; k < kernel_size; ++k) {
            int i_in = start + k;
            if (i_in >= 0 && i_in < in_len) {
                sum += input[((long long)b * channels + c) * in_len + i_in];
                count++;
            } else if (count_include_pad) {
                count++;
            }
        }

        if (count_include_pad) {
            output[idx] = sum / (float)kernel_size;
        } else {
            output[idx] = (count > 0) ? (sum / (float)count) : 0.0f;
        }
    }
}

void avg_pool1d_forward(
    torch::Tensor input,
    torch::Tensor output,
    int kernel_size,
    int stride,
    int padding,
    bool ceil_mode,
    bool count_include_pad)
{
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int in_len = input.size(2);
    const int out_len = output.size(2);
    const int total = batch_size * channels * out_len;

    const int block_size = 256;
    const int grid_size = (total + block_size - 1) / block_size;

    avg_pool1d_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, channels, in_len, out_len,
        kernel_size, stride, padding, count_include_pad
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void avg_pool1d_forward(
    torch::Tensor input,
    torch::Tensor output,
    int kernel_size,
    int stride,
    int padding,
    bool ceil_mode,
    bool count_include_pad);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("avg_pool1d_forward", &avg_pool1d_forward, "Custom 1D average pooling forward kernel");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='avg_pool1d_cuda',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
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
    # Ensure input is on GPU and correct float32 dtype
    x = x.to(device='cuda', dtype=torch.float32)

    batch_size, in_channels, input_length = x.shape

    # Calculate output dimension based on PyTorch formula
    if avg_pool_ceil_mode:
        out_length = (input_length + 2 * avg_pool_padding - avg_pool_kernel_size + avg_pool_stride - 1) // avg_pool_stride + 1
    else:
        out_length = (input_length + 2 * avg_pool_padding - avg_pool_kernel_size) // avg_pool_stride + 1

    output = torch.empty((batch_size, in_channels, out_length), device='cuda', dtype=torch.float32)

    fused_ext.avg_pool1d_forward(
        x,
        output,
        avg_pool_kernel_size,
        avg_pool_stride,
        avg_pool_padding,
        avg_pool_ceil_mode,
        avg_pool_count_include_pad
    )
    return output
