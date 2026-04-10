# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_053509/code_27.py
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

# Optimized CUDA kernel using 2D block decomposition and better memory access patterns
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <algorithm>

__global__ void max_pool2d_kernel_optimized(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int channels,
    int in_h, int in_w,
    int out_h, int out_w,
    int k_size, int stride, int padding, int dilation
) {
    // Map threads to channels and spatial locations
    // blockDim.x = 16, blockDim.y = 16 (for spatial)
    int n = blockIdx.z / channels;
    int c = blockIdx.z % channels;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int w_out = blockIdx.x * blockDim.x + threadIdx.x;

    if (h_out < out_h && w_out < out_w) {
        int h_start = h_out * stride - padding;
        int w_start = w_out * stride - padding;
        
        float max_val = -1e38; // Representing -infinity effectively
        
        int n_c_offset = (n * channels + c) * in_h;
        
        for (int i = 0; i < k_size; ++i) {
            int h_in = h_start + i * dilation;
            if (h_in >= 0 && h_in < in_h) {
                int row_offset = (n_c_offset + h_in) * in_w;
                for (int j = 0; j < k_size; ++j) {
                    int w_in = w_start + j * dilation;
                    if (w_in >= 0 && w_in < in_w) {
                        float val = input[row_offset + w_in];
                        if (val > max_val) max_val = val;
                    }
                }
            }
        }
        output[((n * channels + c) * out_h + h_out) * out_w + w_out] = max_val;
    }
}

void max_pool2d_forward(
    const torch::Tensor input,
    torch::Tensor output,
    int kernel_size, int stride, int padding, int dilation
) {
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int in_h = input.size(2);
    const int in_w = input.size(3);
    const int out_h = output.size(2);
    const int out_w = output.size(3);

    // 2D grid/blocks for spatial dimensions
    dim3 block(16, 16);
    dim3 grid((out_w + block.x - 1) / block.x, (out_h + block.y - 1) / block.y, batch_size * channels);

    max_pool2d_kernel_optimized<<<grid, block>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        batch_size, channels, in_h, in_w, out_h, out_w,
        kernel_size, stride, padding, dilation
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void max_pool2d_forward(const torch::Tensor input, torch::Tensor output, int k, int s, int p, int d);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("max_pool2d_forward", &max_pool2d_forward);
}
"""

max_pool_ext = load_inline(
    name='max_pool2d_opt',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, maxpool_kernel_size, maxpool_stride, maxpool_padding, maxpool_dilation, maxpool_ceil_mode, maxpool_return_indices):
    ih, iw = x.shape[2], x.shape[3]
    div = maxpool_stride
    
    # Calculate output dims correctly
    if maxpool_ceil_mode:
        oh = ((ih + 2 * maxpool_padding - maxpool_dilation * (maxpool_kernel_size - 1) - 1 + div - 1) // div) + 1
        ow = ((iw + 2 * maxpool_padding - maxpool_dilation * (maxpool_kernel_size - 1) - 1 + div - 1) // div) + 1
    else:
        oh = (ih + 2 * maxpool_padding - maxpool_dilation * (maxpool_kernel_size - 1) - 1) // div + 1
        ow = (iw + 2 * maxpool_padding - maxpool_dilation * (maxpool_kernel_size - 1) - 1) // div + 1
    
    output = torch.empty((x.shape[0], x.shape[1], oh, ow), device=x.device, dtype=x.dtype)
    max_pool_ext.max_pool2d_forward(x, output, maxpool_kernel_size, maxpool_stride, maxpool_padding, maxpool_dilation)
    
    return (output, torch.empty(0)) if maxpool_return_indices else output
