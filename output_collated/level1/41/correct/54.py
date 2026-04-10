# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_042936/code_10.py
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

# CUDA kernel with shared memory optimization for maxpool1d
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <algorithm>

#define MAX_SHARED_MEMORY 48000  // 48KB shared memory limit

__global__ void maxpool1d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int features,
    int input_length,
    int output_length,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    // Shared memory for input tile
    extern __shared__ float shared_input[];
    
    // Thread block configuration
    const int threads_per_feature = blockDim.y;
    const int features_per_block = blockDim.y;
    const int positions_per_thread = blockDim.x;
    
    // Feature and batch indices
    const int batch_idx = blockIdx.y / features;
    const int feature_idx = blockIdx.y % features;
    const int feature_in_block = threadIdx.y;
    
    // Calculate tile boundaries
    const int tile_start_output = blockIdx.x * positions_per_thread;
    const int tile_end_output = min(tile_start_output + positions_per_thread, output_length);
    const int tile_output_size = tile_end_output - tile_start_output;
    
    if (tile_output_size <= 0) return;
    
    // Compute input range needed for this tile
    const int start_pos_first = tile_start_output * stride - padding;
    const int start_pos_last = (tile_end_output - 1) * stride - padding;
    const int input_start = max(0, start_pos_first);
    const int input_end = min(input_length, start_pos_last + (kernel_size - 1) * dilation + 1);
    const int input_tile_size = input_end - input_start;
    
    if (input_tile_size <= 0) return;
    
    // Shared memory offset for this feature
    const int shared_offset = feature_in_block * (input_tile_size + (kernel_size - 1) * dilation);
    
    // Load input data into shared memory cooperatively
    for (int i = threadIdx.x; i < input_tile_size; i += positions_per_thread) {
        const int input_idx = input_start + i;
        const int global_idx = (batch_idx * features + feature_idx) * input_length + input_idx;
        shared_input[shared_offset + i] = (input_idx >= 0 && input_idx < input_length) ? input[global_idx] : -3.40282e38f;
    }
    
    __syncthreads();
    
    // Each thread processes one output position
    const int output_pos = tile_start_output + threadIdx.x;
    if (output_pos < tile_end_output) {
        const int start_pos = output_pos * stride - padding;
        const int shared_start = start_pos - input_start;
        
        float max_val = -3.40282e38f;
        
        #pragma unroll
        for (int k = 0; k < kernel_size; ++k) {
            const int pos = shared_start + k * dilation;
            if (pos >= 0 && pos < input_tile_size) {
                const float val = shared_input[shared_offset + pos];
                if (val > max_val) max_val = val;
            }
        }
        
        const int output_idx = (batch_idx * features + feature_idx) * output_length + output_pos;
        output[output_idx] = max_val;
    }
}

void maxpool1d_cuda(
    torch::Tensor input,
    torch::Tensor output,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    const int batch_size = input.size(0);
    const int features = input.size(1);
    const int input_length = input.size(2);
    const int output_length = output.size(2);
    
    // Thread block configuration
    const int threads_per_feature = 8;
    const int positions_per_thread = 32;
    const int threads_x = positions_per_thread;
    const int threads_y = threads_per_feature;
    
    // Grid configuration
    const int blocks_x = (output_length + positions_per_thread - 1) / positions_per_thread;
    const int blocks_y = batch_size * features;
    
    dim3 threads(threads_x, threads_y);
    dim3 blocks(blocks_x, blocks_y);
    
    // Calculate shared memory size
    const int max_input_tile_size = positions_per_thread * stride + (kernel_size - 1) * dilation;
    const int shared_mem_size = threads_per_feature * (max_input_tile_size + (kernel_size - 1) * dilation) * sizeof(float);
    
    maxpool1d_kernel<<<blocks, threads, shared_mem_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        features,
        input_length,
        output_length,
        kernel_size,
        stride,
        padding,
        dilation
    );
}
"""

cpp_source = r"""
void maxpool1d_cuda(torch::Tensor input, torch::Tensor output, int kernel_size, int stride, int padding, int dilation);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("maxpool1d", &maxpool1d_cuda, "Optimized MaxPool1d CUDA with Shared Memory");
}
"""

# Compile extension
maxpool_ext = load_inline(
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
        raise NotImplementedError("Indices not supported in custom kernel")
    
    x = x.contiguous()
    batch, feat, seq_len = x.shape
    
    # Standard formula for MaxPool output size
    if maxpool_ceil_mode:
        out_len = (seq_len + 2 * maxpool_padding - maxpool_dilation * (maxpool_kernel_size - 1) - 1 + maxpool_stride - 1) // maxpool_stride + 1
    else:
        out_len = (seq_len + 2 * maxpool_padding - maxpool_dilation * (maxpool_kernel_size - 1) - 1) // maxpool_stride + 1
        
    output = torch.empty((batch, feat, out_len), device=x.device, dtype=x.dtype)
    
    maxpool_ext.maxpool1d(
        x, output, maxpool_kernel_size, maxpool_stride, maxpool_padding, maxpool_dilation
    )
    
    return output

# Inputs/Constants preserved for compatibility
batch_size = 64
features = 192
sequence_length = 65536
kernel_size = 8
stride = 1
padding = 4
dilation = 3
return_indices = False

def get_init_inputs():
    return [kernel_size, stride, padding, dilation, return_indices]

def get_inputs():
    return [torch.rand(batch_size, features, sequence_length).cuda()]
