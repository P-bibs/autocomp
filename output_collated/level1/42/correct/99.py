# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_062618/code_16.py
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

# -------------------------------------------------------------------------
# CUDA kernel – High throughput direct global memory access
# -------------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>

__global__ void max_pool2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch, int channels,
    int in_h, int in_w,
    int out_h, int out_w,
    int k_size, int stride, int padding) {

    // Indexing for (N, C, H, W)
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output = batch * channels * out_h * out_w;

    if (out_idx >= total_output) return;

    // Decode thread index to N, C, OH, OW
    int tmp = out_idx;
    int ow = tmp % out_w; tmp /= out_w;
    int oh = tmp % out_h; tmp /= out_h;
    int c = tmp % channels; tmp /= channels;
    int b = tmp;

    const int ih_start = oh * stride - padding;
    const int iw_start = ow * stride - padding;
    
    // Pointer to the start of the current channel in input
    const float* input_ptr = input + (b * channels + c) * in_h * in_w;
    
    float max_val = -FLT_MAX;
    
    // Bounds: strictly clamp pooling area to avoid reading outside
    int h_end = min(ih_start + k_size, in_h);
    int w_end = min(iw_start + k_size, in_w);
    int h_begin = max(ih_start, 0);
    int w_begin = max(iw_start, 0);

    for (int i = h_begin; i < h_end; ++i) {
        int row_ptr = i * in_w;
        for (int j = w_begin; j < w_end; ++j) {
            float val = input_ptr[row_ptr + j];
            if (val > max_val) max_val = val;
        }
    }
    
    output[out_idx] = max_val;
}

void launch_max_pool2d(const torch::Tensor& input, torch::Tensor& output, 
                       int k_size, int stride, int padding) {
    int batch = input.size(0);
    int channels = input.size(1);
    int h_out = output.size(2);
    int w_out = output.size(3);
    
    int total_elements = batch * channels * h_out * w_out;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    max_pool2d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        batch, channels, input.size(2), input.size(3),
        h_out, w_out, k_size, stride, padding
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

module = load_inline(
    name='max_pool2d_optimized',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, maxpool_kernel_size, maxpool_stride, maxpool_padding, maxpool_dilation, maxpool_ceil_mode, maxpool_return_indices):
    if maxpool_dilation != 1 or maxpool_ceil_mode or maxpool_return_indices:
        raise NotImplementedError("Only basic MaxPool2D supported.")
    
    h_in, w_in = x.shape[2], x.shape[3]
    h_out = (h_in + 2 * maxpool_padding - maxpool_kernel_size) // maxpool_stride + 1
    w_out = (w_in + 2 * maxpool_padding - maxpool_kernel_size) // maxpool_stride + 1
    
    output = torch.empty((x.shape[0], x.shape[1], h_out, w_out), device=x.device, dtype=x.dtype)
    module.forward(x.contiguous(), output, maxpool_kernel_size, maxpool_stride, maxpool_padding)
    return output
