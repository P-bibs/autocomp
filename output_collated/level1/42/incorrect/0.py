# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_052025/code_5.py
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

# CUDA Kernel for Max Pooling
# Each thread calculates one element of the output tensor.
# By using __restrict__ and standard indexing, we enable the compiler to
# optimize global memory access patterns.
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <algorithm>

__global__ void max_pool2d_kernel(const float* __restrict__ input, float* __restrict__ output, 
                                  int n, int c, int h, int w, 
                                  int k, int s, int p, 
                                  int out_h, int out_w) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = n * c * out_h * out_w;

    if (idx < total_elements) {
        int w_out = idx % out_w;
        int rest = idx / out_w;
        int h_out = rest % out_h;
        int nc = rest / out_h;

        float max_val = -3.402823466e+38f; // -FLT_MAX
        int h_start = h_out * s - p;
        int w_start = w_out * s - p;
        
        int h_end = h_start + k;
        int w_end = w_start + k;

        // Clamp to handle boundaries
        int h_start_clamped = max(h_start, 0);
        int w_start_clamped = max(w_start, 0);
        int h_end_clamped = min(h_end, h);
        int w_end_clamped = min(w_end, w);

        const float* input_ptr = input + (nc * h * w);

        for (int i = h_start_clamped; i < h_end_clamped; ++i) {
            for (int j = w_start_clamped; j < w_end_clamped; ++j) {
                float val = input_ptr[i * w + j];
                if (val > max_val) {
                    max_val = val;
                }
            }
        }
        output[idx] = max_val;
    }
}

void max_pool2d_cuda(torch::Tensor input, torch::Tensor output, int k, int s, int p) {
    int n = input.size(0);
    int c = input.size(1);
    int h = input.size(2);
    int w = input.size(3);
    int out_h = output.size(2);
    int out_w = output.size(3);
    
    int total_elements = n * c * out_h * out_w;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    max_pool2d_kernel<<<blocks, threads>>>(input.data_ptr<float>(), output.data_ptr<float>(), 
                                           n, c, h, w, k, s, p, out_h, out_w);
}
"""

cpp_source = r"""
#include <torch/extension.h>

void max_pool2d_cuda(torch::Tensor input, torch::Tensor output, int k, int s, int p);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("max_pool2d", &max_pool2d_cuda, "Max Pool 2D CUDA kernel");
}
"""

# Compile the extension
max_pool_ext = load_inline(
    name='max_pool_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, maxpool_kernel_size, maxpool_stride, maxpool_padding, maxpool_dilation=1, maxpool_ceil_mode=False, maxpool_return_indices=False):
    # Constraint enforcement based on the goal: use custom kernel
    if maxpool_dilation != 1 or maxpool_ceil_mode or maxpool_return_indices:
        raise NotImplementedError("Custom kernel only supports standard pooling (stride, padding, k size).")
    
    n, c, h, w = x.shape
    out_h = (h + 2 * maxpool_padding - maxpool_kernel_size) // maxpool_stride + 1
    out_w = (w + 2 * maxpool_padding - maxpool_kernel_size) // maxpool_stride + 1
    
    output = torch.empty({n, c, out_h, out_w}, device=x.device, dtype=x.dtype)
    max_pool_ext.max_pool2d(x.contiguous(), output, maxpool_kernel_size, maxpool_stride, maxpool_padding)
    return output
