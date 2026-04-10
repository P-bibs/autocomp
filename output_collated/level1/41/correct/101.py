# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_045555/code_29.py
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

# ----------------------------------------------------------------------
# CUDA kernel – tiled max‑pool 1D with shared‑memory caching
# Optimized to minimize global memory access and bank conflicts.
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <algorithm>

__global__ void maxpool1d_tiled_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int channels,
    const int input_length,
    const int output_length,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation,
    const int tile_width
) {
    // Shared memory: stores the slice of input required for one tile of output
    extern __shared__ float sdata[];

    int tile_idx    = blockIdx.z;      
    int batch_idx   = blockIdx.y;
    int channel_idx = blockIdx.x;

    int tile_start  = tile_idx * tile_width;
    int tile_end    = std::min(output_length, tile_start + tile_width);

    // Calculate source range in global memory for this tile
    int global_start = tile_start * stride - padding;
    int global_end   = (tile_end - 1) * stride - padding + (kernel_size - 1) * dilation + 1;
    
    // Bounds for copying to shared memory
    int load_start = std::max(0, global_start);
    int load_end   = std::min(input_length, global_end);
    int num_to_load = std::max(0, load_end - load_start);

    // Pointer arithmetic
    const float* in_ptr  = input  + (batch_idx * channels + channel_idx) * input_length;
    float*       out_ptr = output + (batch_idx * channels + channel_idx) * output_length;

    // Load into sdata, padding negative indices with -inf
    // We shift sdata so that sdata[0] corresponds to index 'global_start'
    int left_pad = (global_start < 0) ? -global_start : 0;
    
    // Initialize shared memory padding
    for (int i = threadIdx.x; i < left_pad; i += blockDim.x) {
        sdata[i] = -3.402823466e+38F;
    }
    
    // Load input data
    for (int i = threadIdx.x; i < num_to_load; i += blockDim.x) {
        sdata[left_pad + i] = in_ptr[load_start + i];
    }
    
    // Initialize right padding if necessary
    int total_range = (tile_end - 1) * stride - padding + (kernel_size - 1) * dilation + 1 - global_start;
    for (int i = threadIdx.x + left_pad + num_to_load; i < total_range; i += blockDim.x) {
        sdata[i] = -3.402823466e+38F;
    }

    __syncthreads();

    // Perform Pooling
    for (int out_pos = tile_start + threadIdx.x; out_pos < tile_end; out_pos += blockDim.x) {
        int start_pos = out_pos * stride - padding - global_start;
        float max_val = -3.402823466e+38F;
        
        #pragma unroll
        for (int k = 0; k < kernel_size; ++k) {
            max_val = fmaxf(max_val, sdata[start_pos + k * dilation]);
        }
        out_ptr[out_pos] = max_val;
    }
}

void maxpool1d_tiled_forward(
    const torch::Tensor& input,
    torch::Tensor& output,
    int k, int s, int p, int d)
{
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int input_length = input.size(2);
    const int output_length = output.size(2);
    
    const int tile_width = 256; 
    const int tile_count = (output_length + tile_width - 1) / tile_width;
    
    // Max size needed: (tile_width-1)*stride + kernel_size*dilation
    const int max_shared = (tile_width * s) + (k * d) + 1;
    
    dim3 grid(channels, batch_size, tile_count);
    maxpool1d_tiled_kernel<<<grid, 256, max_shared * sizeof(float)>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        channels, input_length, output_length, k, s, p, d, tile_width
    );
}
"""

cpp_source = r"""
void maxpool1d_tiled_forward(const torch::Tensor& input, torch::Tensor& output, int k, int s, int p, int d);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("maxpool1d", &maxpool1d_tiled_forward, "Tiled MaxPool1D");
}
"""

fused_ext = load_inline(
    name='maxpool1d_tiled_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, maxpool_kernel_size, maxpool_stride, maxpool_padding, maxpool_dilation, maxpool_ceil_mode, maxpool_return_indices):
    total_len = x.size(2)
    # Output length calculation
    if maxpool_ceil_mode:
        output_length = ((total_len + 2 * maxpool_padding - maxpool_dilation * (maxpool_kernel_size - 1) - 1) + maxpool_stride - 1) // maxpool_stride + 1
    else:
        output_length = (total_len + 2 * maxpool_padding - maxpool_dilation * (maxpool_kernel_size - 1) - 1) // maxpool_stride + 1
    
    x_gpu = x.cuda().contiguous()
    output = torch.empty((x.size(0), x.size(1), output_length), device='cuda', dtype=x.dtype)
    
    fused_ext.maxpool1d(x_gpu, output, maxpool_kernel_size, maxpool_stride, maxpool_padding, maxpool_dilation)
    return output
