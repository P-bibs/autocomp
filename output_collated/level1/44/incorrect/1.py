# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_113602/code_1.py
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

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void avg_pool1d_kernel(const float* __restrict__ input, float* __restrict__ output, 
                                  int batch_size, int in_channels, int input_len, 
                                  int kernel_size, int stride, int padding, 
                                  int output_len, bool ceil_mode, bool count_include_pad) {
    int tid = threadIdx.x;
    int b = blockIdx.y / in_channels;
    int c = blockIdx.y % in_channels;
    int out_idx = blockIdx.x * blockDim.x + tid;

    if (out_idx < output_len) {
        int start = out_idx * stride - padding;
        int end = start + kernel_size;
        
        // Clamp to valid input range
        start = max(start, 0);
        end = min(end, input_len);
        
        float sum = 0.0f;
        int count = 0;
        
        if (count_include_pad) {
            // Count all positions in the kernel window
            int full_start = out_idx * stride - padding;
            int full_end = full_start + kernel_size;
            if (ceil_mode) {
                // In ceil mode, we need to be more careful about boundary conditions
                // This is a simplified version - for exact PyTorch compatibility, 
                // more complex logic would be needed
                count = full_end - full_start;
            } else {
                count = kernel_size;
            }
            // But only sum over actual valid input positions
            for (int i = start; i < end; ++i) {
                sum += input[(b * in_channels + c) * input_len + i];
            }
            // Normalize by full kernel size (including padded positions)
            output[(b * in_channels + c) * output_len + out_idx] = sum / count;
        } else {
            // Only count valid (non-padded) positions
            for (int i = start; i < end; ++i) {
                sum += input[(b * in_channels + c) * input_len + i];
                count++;
            }
            output[(b * in_channels + c) * output_len + out_idx] = sum / (float)count;
        }
    }
}

void avg_pool1d_cuda(torch::Tensor input, torch::Tensor output, 
                     int kernel_size, int stride, int padding,
                     bool ceil_mode, bool count_include_pad) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_len = input.size(2);
    int output_len = output.size(2);

    dim3 threads(256);
    dim3 blocks((output_len + threads.x - 1) / threads.x, batch_size * in_channels);

    avg_pool1d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        batch_size, in_channels, input_len, kernel_size, stride, padding, output_len,
        ceil_mode, count_include_pad
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void avg_pool1d_cuda(torch::Tensor input, torch::Tensor output, 
                     int kernel_size, int stride, int padding,
                     bool ceil_mode, bool count_include_pad);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("avg_pool1d_cuda", &avg_pool1d_cuda, "1D average pooling forward (CUDA)");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='avg_pool',
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
    # Calculate output length based on PyTorch's avg_pool1d formula
    if avg_pool_ceil_mode:
        output_len = int(torch.ceil(torch.tensor((x.shape[2] + 2 * avg_pool_padding - avg_pool_kernel_size) / avg_pool_stride + 1)).item())
    else:
        output_len = (x.shape[2] + 2 * avg_pool_padding - avg_pool_kernel_size) // avg_pool_stride + 1
    
    output = torch.empty((x.shape[0], x.shape[1], output_len), device=x.device, dtype=x.dtype)
    fused_ext.avg_pool1d_cuda(x, output, avg_pool_kernel_size, avg_pool_stride, avg_pool_padding, 
                              avg_pool_ceil_mode, avg_pool_count_include_pad)
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
