# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_115947/code_1.py
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

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void avg_pool1d_kernel(
    const float* __restrict__ input, 
    float* __restrict__ output,
    int batch_size, int in_channels, int in_len,
    int kernel_size, int stride, int padding,
    int out_len) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * in_channels * out_len) return;

    int cur_out_idx = idx % out_len;
    int cur_channel_idx = (idx / out_len) % in_channels;
    int cur_batch_idx = idx / (out_len * in_channels);

    int start_in = cur_out_idx * stride - padding;
    int end_in = start_in + kernel_size;

    float sum = 0.0f;
    int count = 0;

    for (int i = start_in; i < end_in; ++i) {
        if (i >= 0 && i < in_len) {
            sum += input[cur_batch_idx * (in_channels * in_len) + cur_channel_idx * in_len + i];
            count++;
        }
    }
    output[idx] = sum / (float)count;
}

void avg_pool1d_cuda(torch::Tensor input, torch::Tensor output, int kernel_size, int stride, int padding) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_len = input.size(2);
    int out_len = output.size(2);
    int total_elements = batch_size * in_channels * out_len;

    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    avg_pool1d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        batch_size, in_channels, in_len,
        kernel_size, stride, padding, out_len
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void avg_pool1d_cuda(torch::Tensor input, torch::Tensor output, int kernel_size, int stride, int padding);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("avg_pool1d_cuda", &avg_pool1d_cuda, "1D average pooling CUDA implementation");
}
"""

fused_ext = load_inline(
    name='avg_pool_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, avg_pool_kernel_size, avg_pool_stride, avg_pool_padding, avg_pool_ceil_mode, avg_pool_count_include_pad):
    # Calculate output length (matching PyTorch's floor behavior when ceil_mode=False)
    if avg_pool_ceil_mode:
        out_len = (x.shape[2] + 2 * avg_pool_padding - avg_pool_kernel_size + avg_pool_stride - 1) // avg_pool_stride + 1
    else:
        out_len = (x.shape[2] + 2 * avg_pool_padding - avg_pool_kernel_size) // avg_pool_stride + 1
    
    output = torch.empty((x.shape[0], x.shape[1], out_len), device=x.device, dtype=x.dtype)
    
    fused_ext.avg_pool1d_cuda(x.contiguous(), output, avg_pool_kernel_size, avg_pool_stride, avg_pool_padding)
    return output

# Inputs as defined in the prompt
batch_size, in_channels, input_length = 64, 128, 65536
kernel_size, stride, padding = 8, 1, 4

def get_init_inputs():
    return [kernel_size, stride, padding]

def get_inputs():
    return [torch.rand(batch_size, in_channels, input_length).cuda()]
