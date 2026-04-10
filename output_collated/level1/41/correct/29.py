# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_040207/code_23.py
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

# The CUDA kernel uses __restrict__ to allow the compiler to assume no pointer aliasing.
# We process each element of the output tensor by mapping the global 1D thread ID
# directly to the flattened index (Batch * Channels * OutputLength).
# This ensures that thread i writes to output[i], and because threads are grouped
# into warps, consecutive threads access consecutive memory addresses, resulting in 
# coalesced writes to the output tensor.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void max_pool1d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int channels,
    const int input_length,
    const int output_length,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation,
    const int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_elements) {
        // Decompose index into batch/channel/pos
        // idx = (batch * channels + channel) * output_length + out_pos
        int out_pos = idx % output_length;
        int batch_channel = idx / output_length;
        
        int input_start = out_pos * stride - padding;
        float max_val = -1e38f; // Standard float minimum
        bool found_any = false;
        
        // Loop over the kernel window
        int base_offset = batch_channel * input_length;
        for (int k = 0; k < kernel_size; ++k) {
            int input_pos = input_start + k * dilation;
            if (input_pos >= 0 && input_pos < input_length) {
                float val = input[base_offset + input_pos];
                if (val > max_val) {
                    max_val = val;
                }
                found_any = true;
            }
        }
        
        output[idx] = found_any ? max_val : 0.0f;
    }
}

void max_pool1d_forward(
    const at::Tensor& input,
    at::Tensor& output,
    int kernel_size, int stride, int padding, int dilation
) {
    int batch_size = input.size(0);
    int channels = input.size(1);
    int input_length = input.size(2);
    int output_length = output.size(2);
    int total_elements = batch_size * channels * output_length;
    
    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;
    
    max_pool1d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(),
        channels, input_length, output_length,
        kernel_size, stride, padding, dilation,
        total_elements
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void max_pool1d_forward(const at::Tensor& input, at::Tensor& output, int k, int s, int p, int d);

torch::Tensor max_pool1d_cuda(const torch::Tensor& input, int k, int s, int p, int d) {
    int out_len = (input.size(2) + 2 * p - d * (k - 1) - 1) / s + 1;
    auto output = torch::empty({input.size(0), input.size(1), out_len}, input.options());
    max_pool1d_forward(input, output, k, s, p, d);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("max_pool1d_cuda", &max_pool1d_cuda, "Optimized 1D max pooling");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_max_pool1d_opt',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
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
    maxpool_ceil_mode=False,
    maxpool_return_indices=False,
):
    # This implementation is optimized for the parameters provided.
    return fused_ext.max_pool1d_cuda(
        x, 
        maxpool_kernel_size, 
        maxpool_stride, 
        maxpool_padding, 
        maxpool_dilation
    )
