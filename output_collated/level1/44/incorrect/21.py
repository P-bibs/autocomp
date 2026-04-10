# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_115226/code_0.py
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void avg_pool1d_kernel(
    const float* input,
    float* output,
    int batch_size,
    int in_channels,
    int input_length,
    int output_length,
    int kernel_size,
    int stride,
    int padding,
    bool ceil_mode,
    bool count_include_pad
) {
    int batch_idx = blockIdx.x;
    int channel_idx = blockIdx.y;
    int output_idx = blockIdx.z * blockDim.x + threadIdx.x;
    
    if (batch_idx >= batch_size || channel_idx >= in_channels || output_idx >= output_length) {
        return;
    }
    
    int input_start = output_idx * stride - padding;
    int input_end = input_start + kernel_size;
    
    float sum = 0.0f;
    int valid_count = 0;
    
    for (int i = input_start; i < input_end; i++) {
        if (i >= 0 && i < input_length) {
            int input_idx = batch_idx * in_channels * input_length + channel_idx * input_length + i;
            sum += input[input_idx];
            valid_count++;
        } else if (count_include_pad) {
            valid_count++;
        }
    }
    
    float result = (valid_count > 0) ? (sum / valid_count) : 0.0f;
    
    int output_idx_flat = batch_idx * in_channels * output_length + channel_idx * output_length + output_idx;
    output[output_idx_flat] = result;
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
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_length = input.size(2);
    
    int output_length;
    if (ceil_mode) {
        output_length = (int)ceilf((float)(input_length + 2 * padding - kernel_size) / stride) + 1;
    } else {
        output_length = (input_length + 2 * padding - kernel_size) / stride + 1;
    }
    
    dim3 block_size(256);
    dim3 grid_size(batch_size, in_channels, (output_length + block_size.x - 1) / block_size.x);
    
    avg_pool1d_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
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
    m.def("avg_pool1d_forward", &avg_pool1d_forward, "Average Pooling 1D forward");
}
"""

fused_ext = load_inline(
    name='avg_pool1d_ext',
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
    # Calculate output length
    input_length = x.size(2)
    if avg_pool_ceil_mode:
        output_length = int(torch.ceil(torch.tensor((input_length + 2 * avg_pool_padding - avg_pool_kernel_size) / avg_pool_stride))) + 1
    else:
        output_length = (input_length + 2 * avg_pool_padding - avg_pool_kernel_size) // avg_pool_stride + 1
    
    # Create output tensor
    output = torch.empty(x.size(0), x.size(1), output_length, dtype=x.dtype, device=x.device)
    
    # Call custom CUDA kernel
    fused_ext.avg_pool1d_forward(
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
