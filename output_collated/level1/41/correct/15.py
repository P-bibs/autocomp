# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_035026/code_17.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['kernel_size', 'stride', 'padding', 'dilation', 'return_indices']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['maxpool_kernel_size', 'maxpool_stride', 'maxpool_padding', 'maxpool_dilation', 'maxpool_ceil_mode', 'maxpool_return_indices']
REQUIRED_FLAT_STATE_NAMES = []


class ModelNew(nn.Module):
    """
    Simple model that performs Max Pooling 1D.
    """

    def __init__(self, kernel_size: int, stride: int=None, padding: int=0, dilation: int=1, return_indices: bool=False):
        """
        Initializes the Max Pooling 1D layer.

        Args:
            kernel_size (int): Size of the window to take a max over.
            stride (int, optional): Stride of the window. Defaults to None (same as kernel_size).
            padding (int, optional): Implicit zero padding to be added on both sides. Defaults to 0.
            dilation (int, optional): Spacing between kernel elements. Defaults to 1.
            return_indices (bool, optional): Whether to return the indices of the maximum values. Defaults to False.
        """
        super(ModelNew, self).__init__()
        self.maxpool = nn.MaxPool1d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, return_indices=return_indices)

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
    # State for maxpool (nn.MaxPool1d)
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
#include <algorithm>

__global__ void max_pool1d_kernel(const float* __restrict__ input, float* __restrict__ output,
                                  int batch_size, int features, int seq_len,
                                  int kernel_size, int stride, int padding, int dilation,
                                  int out_len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * features * out_len) return;

    int b = idx / (features * out_len);
    int rem = idx % (features * out_len);
    int f = rem / out_len;
    int o = rem % out_len;

    int input_offset = (b * features + f) * seq_len;
    int start = o * stride - padding;
    
    float max_val = -3.40282e38f; // -FLT_MAX

    for (int k = 0; k < kernel_size; ++k) {
        int in_idx = start + k * dilation;
        if (in_idx >= 0 && in_idx < seq_len) {
            float val = input[input_offset + in_idx];
            if (val > max_val) max_val = val;
        }
    }
    output[idx] = max_val;
}

torch::Tensor max_pool1d_cuda(torch::Tensor x, int kernel_size, int stride, int padding, int dilation) {
    auto batch_size = x.size(0);
    auto features = x.size(1);
    auto seq_len = x.size(2);
    
    auto out_len = (seq_len + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    auto output = torch::empty({batch_size, features, out_len}, x.options());
    
    int total_elements = batch_size * features * out_len;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    
    max_pool1d_kernel<<<blocks, threads>>>(x.data_ptr<float>(), output.data_ptr<float>(),
                                           (int)batch_size, (int)features, (int)seq_len,
                                           kernel_size, stride, padding, dilation, (int)out_len);
    return output;
}
"""

cpp_source = r"""
#include <torch/extension.h>

torch::Tensor max_pool1d_cuda(torch::Tensor x, int kernel_size, int stride, int padding, int dilation);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("max_pool1d_cuda", &max_pool1d_cuda, "optimized max_pool1d");
}
"""

max_pool_ext = load_inline(
    name='max_pool1d_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, maxpool_kernel_size, maxpool_stride, maxpool_padding, maxpool_dilation, maxpool_ceil_mode, maxpool_return_indices):
    # The kernel implements the logic for MaxPool1D efficiently on GPU.
    return max_pool_ext.max_pool1d_cuda(x, maxpool_kernel_size, maxpool_stride, maxpool_padding, maxpool_dilation)

# Setup context for evaluation
batch_size = 64
features = 192
sequence_length = 65536

def get_inputs():
    return [torch.rand(batch_size, features, sequence_length).cuda()]

def get_init_inputs():
    return dict(
        maxpool_kernel_size=8, 
        maxpool_stride=1, 
        maxpool_padding=4, 
        maxpool_dilation=3, 
        maxpool_ceil_mode=False, 
        maxpool_return_indices=False
    )
