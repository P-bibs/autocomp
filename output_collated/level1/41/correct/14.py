# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_035026/code_16.py
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
#include <vector_types.h>
#include <vector_functions.h>

__global__ void max_pool1d_dilated_vectorized_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int channels,
    const int input_length,
    const int output_length,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_ch = blockIdx.y; 
    
    if (out_idx >= output_length) return;

    const float* input_ptr = input + (batch_ch * input_length);
    float* output_ptr = output + (batch_ch * output_length);

    int in_start = out_idx * stride - padding;
    float max_val = -1e30f; // Sufficiently small for float
    
    // Unrolled loop for kernel accumulation
    #pragma unroll
    for (int k = 0; k < kernel_size; ++k) {
        int in_pos = in_start + k * dilation;
        if (in_pos >= 0 && in_pos < input_length) {
            float val = input_ptr[in_pos];
            if (val > max_val) max_val = val;
        }
    }
    output_ptr[out_idx] = max_val;
}

void max_pool1d_dilated_forward(
    const torch::Tensor input,
    torch::Tensor output,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation
) {
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int output_length = output.size(2);
    
    dim3 threads(256);
    // Flatten Batch and Channels for better occupancy
    dim3 blocks((output_length + 255) / 256, batch_size * channels);
    
    max_pool1d_dilated_vectorized_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(),
        channels, input.size(2), output_length, 
        kernel_size, stride, padding, dilation
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void max_pool1d_dilated_forward(const torch::Tensor input, torch::Tensor output, const int kernel_size, const int stride, const int padding, const int dilation);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("max_pool1d_dilated_forward", &max_pool1d_dilated_forward, "Optimized Max Pool1D Backward");
}
"""

module = load_inline(
    name='max_pool1d_optimized',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(
    x,
    *,
    maxpool_kernel_size,
    maxpool_stride,
    maxpool_padding,
    maxpool_dilation,
    maxpool_ceil_mode,
    maxpool_return_indices,
):
    L = x.shape[2]
    num = L + 2 * maxpool_padding - maxpool_dilation * (maxpool_kernel_size - 1) - 1
    if maxpool_ceil_mode:
        output_length = int((num + maxpool_stride - 1) // maxpool_stride + 1)
    else:
        output_length = int(num // maxpool_stride + 1)
    
    output = torch.empty(x.shape[0], x.shape[1], output_length, device=x.device, dtype=x.dtype)
    
    module.max_pool1d_dilated_forward(
        x, output, 
        maxpool_kernel_size, 
        maxpool_stride, 
        maxpool_padding, 
        maxpool_dilation
    )
    
    return (output, None) if maxpool_return_indices else output
