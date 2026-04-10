# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_053509/code_22.py
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
#include <algorithm>

#define TILE_DIM 16
#define MAX_K 7

__global__ void max_pool2d_tiled_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int channels, int in_h, int in_w,
    int out_h, int out_w,
    int k_size, int stride, int padding) {

    // Shared memory: cache input tile for the block
    // Dimensions: (TILE_DIM + k-1) * (TILE_DIM + k-1)
    __shared__ float s_tile[TILE_DIM + MAX_K][TILE_DIM + MAX_K];

    int b = blockIdx.z;
    int c = blockIdx.y;
    int out_y_start = blockIdx.x * TILE_DIM;
    int out_x_start = blockIdx.y * TILE_DIM; // Note: simplified logic

    // Simplified for robustness: use 2D block grid mapping channels/batches to threads
    int b_idx = blockIdx.z;
    int c_idx = blockIdx.y;
    int out_y = blockIdx.x * TILE_DIM + threadIdx.y;
    int out_x = threadIdx.x; 

    if (out_y < out_h) {
        for (int ow = 0; ow < out_w; ++ow) {
            float max_val = -1e38f;
            int ih_base = out_y * stride - padding;
            int iw_base = ow * stride - padding;

            for (int i = 0; i < k_size; ++i) {
                int ih = ih_base + i;
                if (ih >= 0 && ih < in_h) {
                    for (int j = 0; j < k_size; ++j) {
                        int iw = iw_base + j;
                        if (iw >= 0 && iw < in_w) {
                            float val = input[((b_idx * channels + c_idx) * in_h + ih) * in_w + iw];
                            if (val > max_val) max_val = val;
                        }
                    }
                }
            }
            output[(((b_idx * channels + c_idx) * out_h + out_y) * out_w) + ow] = max_val;
        }
    }
}

void max_pool2d_forward_cuda(const torch::Tensor& input, torch::Tensor& output, 
                             int k_size, int stride, int padding) {
    int b = input.size(0), c = input.size(1), h = output.size(2), w = output.size(3);
    dim3 block(16, 16);
    dim3 grid((h + 15) / 16, c, b);
    
    // Launch kernel
    max_pool2d_tiled_kernel<<<grid, block>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        c, input.size(2), input.size(3), h, w, k_size, stride, padding
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void max_pool2d_forward_cuda(const torch::Tensor& input, torch::Tensor& output, int k_size, int stride, int padding);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_forward_cuda, "Optimized MaxPool2d Forward");
}
"""

# Compile the extension
maxpool_ext = load_inline(
    name='maxpool_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, maxpool_kernel_size, maxpool_stride, maxpool_padding, maxpool_dilation, maxpool_ceil_mode, maxpool_return_indices):
    h_in, w_in = x.shape[2], x.shape[3]
    h_out = (h_in + 2 * maxpool_padding - maxpool_kernel_size) // maxpool_stride + 1
    w_out = (w_in + 2 * maxpool_padding - maxpool_kernel_size) // maxpool_stride + 1
    
    output = torch.empty((x.shape[0], x.shape[1], h_out, w_out), device=x.device, dtype=x.dtype)
    
    # Direct call to custom C++/CUDA kernel
    maxpool_ext.forward(x.contiguous(), output, maxpool_kernel_size, maxpool_stride, maxpool_padding)
    
    return output
