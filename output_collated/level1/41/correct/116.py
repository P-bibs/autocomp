# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_050814/code_27.py
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

# The CUDA kernel uses Shared Memory tiling to cache input windows.
# Each thread block loads a chunk of the input sequence into shared memory
# to minimize global memory bandwidth consumption for overlapping windows.

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <algorithm>

#define SHARED_MEM_SIZE 256 

__global__ void maxpool1d_shared_memory_kernel(
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
    extern __shared__ float s_data[];

    int feat_idx = blockIdx.y;
    int block_out_start = blockIdx.x * blockDim.x;

    // Calculate range of input data needed for this block of outputs
    int first_out_idx = block_out_start;
    int last_out_idx = min(block_out_start + blockDim.x - 1, output_length - 1);
    
    int start_input = first_out_idx * stride - padding;
    int end_input = (last_out_idx * stride - padding) + (kernel_size - 1) * dilation;
    
    int tile_start = max(0, start_input);
    int tile_end = min(input_length - 1, end_input);
    int tile_size = tile_end - tile_start + 1;

    // Cooperative loading into shared memory
    for (int i = threadIdx.x; i < tile_size; i += blockDim.x) {
        s_data[i] = input[feat_idx * input_length + (tile_start + i)];
    }
    __syncthreads();

    // Compute max pooling
    int out_idx = block_out_start + threadIdx.x;
    if (out_idx < output_length) {
        float max_val = -1e38f;
        int current_start = out_idx * stride - padding;
        
        for (int k = 0; k < kernel_size; ++k) {
            int pos = current_start + k * dilation;
            if (pos >= tile_start && pos <= tile_end) {
                float val = s_data[pos - tile_start];
                if (val > max_val) max_val = val;
            }
        }
        output[feat_idx * output_length + out_idx] = max_val;
    }
}

void maxpool1d_cuda_interface(torch::Tensor input, torch::Tensor output, int k, int s, int p, int d) {
    int batch = input.size(0);
    int feat = input.size(1);
    int input_len = input.size(2);
    int out_len = output.size(2);
    
    // Using 128 threads for block size
    int threads_per_block = 128;
    int blocks_per_feat = (out_len + threads_per_block - 1) / threads_per_block;
    dim3 grid(blocks_per_feat, batch * feat);
    
    // Shared memory size calculation can be dynamic but we cap at a safe limit
    // Here we define the max possible tile size we might need
    size_t shared_size = (threads_per_block * s + k * d) * sizeof(float);
    
    maxpool1d_shared_memory_kernel<<<grid, threads_per_block, shared_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), 
        feat, input_len, out_len, k, s, p, d
    );
}
"""

cpp_source = r"""
void maxpool1d_cuda_interface(torch::Tensor input, torch::Tensor output, int k, int s, int p, int d);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("maxpool1d", &maxpool1d_cuda_interface, "Optimized MaxPool1d CUDA with Shared Memory");
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
