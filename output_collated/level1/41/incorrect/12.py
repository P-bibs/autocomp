# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_041646/code_9.py
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

# Optimized CUDA kernel with shared memory tiling
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <algorithm>

#define SHARED_MEMORY_SIZE 8192

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
    extern __shared__ float shared_input[];
    
    // Block and thread indices
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int feature_batch_idx = blockIdx.y;
    
    // Calculate batch and feature indices
    int batch_idx = feature_batch_idx / features;
    int feat_idx = feature_batch_idx % features;
    
    // Base offsets for input and output
    int input_base_offset = (batch_idx * features + feat_idx) * input_length;
    int output_base_offset = (batch_idx * features + feat_idx) * output_length;
    
    // Tile size and halo size
    int tile_size = blockDim.x;
    int halo_size = (kernel_size - 1) * dilation;
    
    // Process multiple output elements per block
    for (int tile_start = bid * tile_size; tile_start < output_length; tile_start += gridDim.x * tile_size) {
        // Calculate the range of input positions needed for this tile
        int first_output = tile_start;
        int last_output = min(tile_start + tile_size - 1, output_length - 1);
        
        if (first_output > last_output) break;
        
        // Compute the input range needed
        int start_pos_first = first_output * stride - padding;
        int start_pos_last = last_output * stride - padding;
        int end_pos_last = start_pos_last + (kernel_size - 1) * dilation;
        
        int shared_start = start_pos_first;
        int shared_end = end_pos_last;
        int shared_size = shared_end - shared_start + 1;
        
        // Clamp to valid input range
        int valid_start = max(0, shared_start);
        int valid_end = min(input_length - 1, shared_end);
        
        // Cooperative loading of input data into shared memory
        for (int i = tid; i < shared_size; i += blockDim.x) {
            int input_pos = shared_start + i;
            if (input_pos >= valid_start && input_pos <= valid_end) {
                shared_input[i] = input[input_base_offset + input_pos];
            } else {
                shared_input[i] = -3.40282e38f; // FLT_MIN
            }
        }
        
        __syncthreads();
        
        // Each thread computes one output
        for (int out_idx = tile_start + tid; out_idx <= last_output; out_idx += blockDim.x) {
            int local_out_pos = out_idx - tile_start;
            int start_pos = out_idx * stride - padding;
            
            float max_val = -3.40282e38f; // FLT_MIN
            
            #pragma unroll 4
            for (int k = 0; k < kernel_size; ++k) {
                int pos = start_pos + k * dilation;
                int shared_idx = pos - shared_start;
                
                if (shared_idx >= 0 && shared_idx < shared_size) {
                    float val = shared_input[shared_idx];
                    if (val > max_val) max_val = val;
                }
            }
            
            output[output_base_offset + local_out_pos] = max_val;
        }
        
        __syncthreads();
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
    int batch_size = input.size(0);
    int features = input.size(1);
    int input_length = input.size(2);
    int output_length = output.size(2);
    
    // Launch configuration
    int threads_per_block = 256;
    int blocks_per_grid = min(65535, (output_length + threads_per_block - 1) / threads_per_block);
    
    dim3 threads(threads_per_block);
    dim3 blocks(blocks_per_grid, batch_size * features);
    
    // Shared memory size calculation
    int halo_size = (kernel_size - 1) * dilation;
    int shared_mem_per_block = (threads_per_block + halo_size) * sizeof(float);
    
    // Clamp shared memory to avoid exceeding limits
    if (shared_mem_per_block > 48 * 1024) {
        shared_mem_per_block = 48 * 1024;
    }
    
    maxpool1d_kernel<<<blocks, threads, shared_mem_per_block>>>(
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
