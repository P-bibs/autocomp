# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_042936/code_17.py
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

# The optimized kernel uses dynamic shared memory to cache the input tile 
# required for the output indices assigned to a block. This reduces global
# memory pressure, especially beneficial when kernel_size is large.
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <algorithm>

__global__ void maxpool1d_shared_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int features,
    int input_length,
    int output_length,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    extern __shared__ float tile[];

    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int bf_idx = blockIdx.y;

    // Define the range of input elements this block needs to load
    int start_out = blockIdx.x * blockDim.x;
    int end_out = min(start_out + blockDim.x - 1, output_length - 1);
    
    int tile_start = max(0, start_out * stride - padding);
    int tile_end = min(input_length - 1, (end_out * stride - padding) + (kernel_size - 1) * dilation);
    int tile_len = tile_end - tile_start + 1;

    // Cooperative loading into shared memory
    for (int i = threadIdx.x; i < tile_len; i += blockDim.x) {
        tile[i] = input[bf_idx * input_length + tile_start + i];
    }
    __syncthreads();

    if (out_idx >= output_length) return;

    int start_pos = out_idx * stride - padding;
    float max_val = -3.40282e38f; // FLT_MIN equivalent

    #pragma unroll
    for (int k = 0; k < kernel_size; ++k) {
        int pos = start_pos + k * dilation;
        if (pos >= tile_start && pos <= tile_end) {
            float val = tile[pos - tile_start];
            if (val > max_val) max_val = val;
        }
    }
    output[bf_idx * output_length + out_idx] = max_val;
}

void maxpool1d_cuda(torch::Tensor input, torch::Tensor output, int k, int s, int p, int d) {
    int batch = input.size(0);
    int feat = input.size(1);
    int in_len = input.size(2);
    int out_len = output.size(2);
    
    int threads_per_block = 256;
    int blocks_x = (out_len + threads_per_block - 1) / threads_per_block;
    dim3 blocks(blocks_x, batch * feat);
    
    // Calculate required shared memory size
    // Each thread block processes a range of output indices
    int max_window_extent = (k - 1) * d + 1;
    int tile_len = threads_per_block * s + max_window_extent;
    size_t shared_mem_size = tile_len * sizeof(float);

    maxpool1d_shared_kernel<<<blocks, threads_per_block, shared_mem_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), 
        feat, in_len, out_len, k, s, p, d
    );
}
"""

cpp_source = r"""
void maxpool1d_cuda(torch::Tensor input, torch::Tensor output, int k, int s, int p, int d);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("maxpool1d", &maxpool1d_cuda, "Optimized MaxPool1d CUDA with shared memory");
}
"""

_maxpool_ext = load_inline(
    name='maxpool_ext',
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
    if maxpool_return_indices:
        raise NotImplementedError("Indices not supported")
    
    x = x.contiguous()
    batch, feat, seq_len = x.shape
    
    if maxpool_ceil_mode:
        out_len = (seq_len + 2 * maxpool_padding - maxpool_dilation * (maxpool_kernel_size - 1) - 1 + maxpool_stride - 1) // maxpool_stride + 1
    else:
        out_len = (seq_len + 2 * maxpool_padding - maxpool_dilation * (maxpool_kernel_size - 1) - 1) // maxpool_stride + 1
        
    output = torch.empty((batch, feat, out_len), device=x.device, dtype=x.dtype)
    
    _maxpool_ext.maxpool1d(
        x, output, int(maxpool_kernel_size), int(maxpool_stride), 
        int(maxpool_padding), int(maxpool_dilation)
    )
    
    return output
