# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_054839/code_6.py
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

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void max_pool2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int in_h, int in_w,
    int out_h, int out_w, int k_size, int stride, int padding) {

    int batch_idx = blockIdx.z; // Batch * Channels
    int out_y = blockIdx.y * TILE_SIZE + threadIdx.y;
    int out_x = blockIdx.x * TILE_SIZE + threadIdx.x;

    if (out_y < out_h && out_x < out_w) {
        int in_y_start = out_y * stride - padding;
        int in_x_start = out_x * stride - padding;
        
        float max_val = -3.402823466e+38F; // Approximation of -FLT_MAX
        
        const float* input_ptr = input + (batch_idx * in_h * in_w);
        
        #pragma unroll
        for (int i = 0; i < k_size; ++i) {
            int y = in_y_start + i;
            if (y >= 0 && y < in_h) {
                const float* row_ptr = input_ptr + (y * in_w);
                #pragma unroll
                for (int j = 0; j < k_size; ++j) {
                    int x = in_x_start + j;
                    if (x >= 0 && x < in_w) {
                        float val = row_ptr[x];
                        if (val > max_val) max_val = val;
                    }
                }
            }
        }
        output[(batch_idx * out_h * out_w) + (out_y * out_w + out_x)] = max_val;
    }
}

void max_pool2d_forward(const torch::Tensor& input, torch::Tensor& output, 
                        int k_size, int stride, int padding) {
    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    int out_H = output.size(2);
    int out_W = output.size(3);

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 grid((out_W + TILE_SIZE - 1) / TILE_SIZE, 
              (out_H + TILE_SIZE - 1) / TILE_SIZE, 
              N * C);

    max_pool2d_kernel<<<grid, threads>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        N, C, H, W, out_H, out_W, k_size, stride, padding
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void max_pool2d_forward(const torch::Tensor& input, torch::Tensor& output, int k_size, int stride, int padding);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("max_pool2d_forward", &max_pool2d_forward, "Optimized Max Pool 2D Forward");
}
"""

module = load_inline(
    name='max_pool2d_optimized',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, maxpool_kernel_size, maxpool_stride, maxpool_padding, maxpool_dilation, maxpool_ceil_mode, maxpool_return_indices):
    if maxpool_dilation != 1:
        raise NotImplementedError("Dilation not supported in this optimized version")
    if maxpool_ceil_mode:
        raise NotImplementedError("Ceil mode not supported in this optimized version")
    if maxpool_return_indices:
        raise NotImplementedError("Indices not implemented in this optimized version")

    n, c, h_in, w_in = x.shape
    h_out = (h_in + 2 * maxpool_padding - maxpool_kernel_size) // maxpool_stride + 1
    w_out = (w_in + 2 * maxpool_padding - maxpool_kernel_size) // maxpool_stride + 1
    
    output = torch.empty((n, c, h_out, w_out), device=x.device, dtype=x.dtype)
    module.max_pool2d_forward(x.contiguous(), output, maxpool_kernel_size, maxpool_stride, maxpool_padding)
    return output
