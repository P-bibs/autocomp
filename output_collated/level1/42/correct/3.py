# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_052448/code_4.py
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
from torch.utils.cpp_extension import load_inline

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector_types.h>
#include <cmath>

__global__ void max_pool2d_kernel_vec(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size, int channels,
    int input_height, int input_width,
    int output_height, int output_width,
    int k_size, int stride, int padding, int dilation
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = batch_size * channels * output_height * output_width;
    
    if (tid >= total_outputs) return;

    int w_out = tid % output_width;
    int h_out = (tid / output_width) % output_height;
    int c = (tid / (output_width * output_height)) % channels;
    int n = tid / (output_width * output_height * channels);

    int h_start = h_out * stride - padding;
    int w_start = w_out * stride - padding;
    
    float max_val = -INFINITY;

    for (int h = 0; h < k_size; ++h) {
        int ih = h_start + h * dilation;
        if (ih < 0 || ih >= input_height) continue;
        
        for (int w = 0; w < k_size; ++w) {
            int iw = w_start + w * dilation;
            if (iw >= 0 && iw < input_width) {
                int input_idx = ((n * channels + c) * input_height + ih) * input_width + iw;
                max_val = fmaxf(max_val, input[input_idx]);
            }
        }
    }
    output[tid] = max_val;
}

void max_pool2d_forward(const torch::Tensor input, torch::Tensor output, int ks, int st, int pad, int dil) {
    int total = output.numel();
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    max_pool2d_kernel_vec<<<blocks, threads>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        input.size(0), input.size(1), input.size(2), input.size(3),
        output.size(2), output.size(3), ks, st, pad, dil
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void max_pool2d_forward(const torch::Tensor input, torch::Tensor output, int ks, int st, int pad, int dil);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("max_pool2d_forward", &max_pool2d_forward, "Vectorized Max Pool 2D");
}
"""

max_pool_ext = load_inline(name='max_pool2d_vec', cpp_sources=cpp_source, cuda_sources=cuda_kernel, 
                           extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True)

def functional_model(x, *, maxpool_kernel_size, maxpool_stride, maxpool_padding, maxpool_dilation,
                     maxpool_ceil_mode, maxpool_return_indices):
    h_in, w_in = x.shape[2], x.shape[3]
    
    if maxpool_ceil_mode:
        h_out = int(torch.ceil(torch.tensor((h_in + 2 * maxpool_padding - maxpool_dilation * (maxpool_kernel_size - 1) - 1) / maxpool_stride + 1)).item())
        w_out = int(torch.ceil(torch.tensor((w_in + 2 * maxpool_padding - maxpool_dilation * (maxpool_kernel_size - 1) - 1) / maxpool_stride + 1)).item())
    else:
        h_out = int(torch.floor(torch.tensor((h_in + 2 * maxpool_padding - maxpool_dilation * (maxpool_kernel_size - 1) - 1) / maxpool_stride + 1)).item())
        w_out = int(torch.floor(torch.tensor((w_in + 2 * maxpool_padding - maxpool_dilation * (maxpool_kernel_size - 1) - 1) / maxpool_stride + 1)).item())
    
    output = torch.empty((x.shape[0], x.shape[1], h_out, w_out), device=x.device, dtype=x.dtype)
    max_pool_ext.max_pool2d_forward(x, output, maxpool_kernel_size, maxpool_stride, maxpool_padding, maxpool_dilation)
    
    if maxpool_return_indices:
        return output, torch.empty_like(output, dtype=torch.long)
    else:
        return output
