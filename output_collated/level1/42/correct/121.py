# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_063920/code_21.py
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
#include <float.h>

#define BLOCK_DIM 16

__global__ void max_pool2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int total_channels, 
    int in_h, int in_w,
    int out_h, int out_w,
    int k_size, int stride, int padding) {

    // Map 3D grid to Batch-Channel (z), Y (y), X (x)
    int nc = blockIdx.z;
    int out_y = blockIdx.y * BLOCK_DIM + threadIdx.y;
    int out_x = blockIdx.x * BLOCK_DIM + threadIdx.x;

    if (nc < total_channels && out_y < out_h && out_x < out_w) {
        float max_val = -1e30f; // Use a large finite value
        
        int in_y_base = out_y * stride - padding;
        int in_x_base = out_x * stride - padding;
        
        const float* input_ptr = input + (size_t)nc * in_h * in_w;
        
        // Loop over the pooling window
        #pragma unroll
        for (int i = 0; i < k_size; ++i) {
            int in_y = in_y_base + i;
            if (in_y >= 0 && in_y < in_h) {
                const float* row_ptr = input_ptr + (size_t)in_y * in_w;
                #pragma unroll
                for (int j = 0; j < k_size; ++j) {
                    int in_x = in_x_base + j;
                    if (in_x >= 0 && in_x < in_w) {
                        float val = row_ptr[in_x];
                        if (val > max_val) max_val = val;
                    }
                }
            }
        }
        output[(size_t)nc * out_h * out_w + (size_t)out_y * out_w + out_x] = max_val;
    }
}

void launch_max_pool2d(const torch::Tensor& input, torch::Tensor& output, 
                       int k_size, int stride, int padding) {
    int batch = input.size(0);
    int channels = input.size(1);
    int total_channels = batch * channels;
    int in_h = input.size(2);
    int in_w = input.size(3);
    int out_h = output.size(2);
    int out_w = output.size(3);

    dim3 block(BLOCK_DIM, BLOCK_DIM);
    dim3 grid((out_w + BLOCK_DIM - 1) / BLOCK_DIM,
              (out_h + BLOCK_DIM - 1) / BLOCK_DIM,
              total_channels);

    max_pool2d_kernel<<<grid, block>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        total_channels, in_h, in_w, out_h, out_w, k_size, stride, padding
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void launch_max_pool2d(const torch::Tensor& input, torch::Tensor& output, int k_size, int stride, int padding);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &launch_max_pool2d, "Optimized MaxPool2D Forward");
}
"""

max_pool_module = load_inline(
    name='max_pool2d_optimized',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, maxpool_kernel_size, maxpool_stride, maxpool_padding, maxpool_dilation, maxpool_ceil_mode, maxpool_return_indices):
    if maxpool_dilation != 1 or maxpool_ceil_mode or maxpool_return_indices:
        raise NotImplementedError("Only basic MaxPool2D supported.")
    
    x = x.contiguous()
    h_in, w_in = x.shape[2], x.shape[3]
    h_out = (h_in + 2 * maxpool_padding - maxpool_kernel_size) // maxpool_stride + 1
    w_out = (w_in + 2 * maxpool_padding - maxpool_kernel_size) // maxpool_stride + 1
    
    output = torch.empty((x.shape[0], x.shape[1], h_out, w_out), device=x.device, dtype=x.dtype)
    max_pool_module.forward(x, output, maxpool_kernel_size, maxpool_stride, maxpool_padding)
    return output
