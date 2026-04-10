# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_050814/code_17.py
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

# The implementation below optimizes the 1D MaxPool by using shared memory 
# and a tiled approach to improve memory throughput.

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <algorithm>

#define TILE_SIZE 128
#define MAX_SHARED_MEM 32768 // Safely within constraints for most architectures

__global__ void maxpool1d_shared_kernel(
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
    extern __shared__ float shared_data[];
    
    int batch_idx = blockIdx.y;
    int channel_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    const float* in_ptr = input + (batch_idx * channels + channel_idx) * input_length;
    float* out_ptr = output + (batch_idx * channels + channel_idx) * output_length;
    
    // Each block processes a tile of the output sequence
    int start_out_pos = blockIdx.z * TILE_SIZE;
    int end_out_pos = min(start_out_pos + TILE_SIZE, output_length);
    
    // Bounds check for the input region mapped by this tile
    int start_in_pos = start_out_pos * stride - padding;
    int end_in_pos = ((end_out_pos - 1) * stride - padding) + (kernel_size - 1) * dilation + 1;
    
    // Load input tiles into shared memory
    // Shared memory size is guaranteed to be enough via host-side logic
    int load_start = max(0, start_in_pos);
    int load_end = min(input_length, end_in_pos);
    int num_to_load = load_end - load_start;
    
    for (int i = tid; i < num_to_load; i += blockDim.x) {
        shared_data[i] = in_ptr[load_start + i];
    }
    __syncthreads();
    
    // Compute max
    for (int out_pos = start_out_pos + tid; out_pos < end_out_pos; out_pos += blockDim.x) {
        int window_start = out_pos * stride - padding;
        float max_val = -3.402823466e+38F; 
        
        #pragma unroll
        for (int k = 0; k < kernel_size; ++k) {
            int in_pos = window_start + k * dilation;
            if (in_pos >= load_start && in_pos < load_end) {
                float val = shared_data[in_pos - load_start];
                if (val > max_val) max_val = val;
            }
        }
        out_ptr[out_pos] = max_val;
    }
}

void maxpool1d_forward(const torch::Tensor& input, torch::Tensor& output, int k, int s, int p, int d) {
    int batch_size = input.size(0);
    int channels = input.size(1);
    int input_length = input.size(2);
    int output_length = output.size(2);
    
    // Tiled approach
    int tiles = (output_length + TILE_SIZE - 1) / TILE_SIZE;
    dim3 grid(channels, batch_size, tiles);
    int threads = 128;
    
    // Calculate required shared memory based on input span per block
    // Max span = (TILE_SIZE-1)*stride + (k-1)*dilation + 1
    int shared_mem = (TILE_SIZE * s + k * d) * sizeof(float);
    
    maxpool1d_shared_kernel<<<grid, threads, shared_mem>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(),
        channels, input_length, output_length, k, s, p, d
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void maxpool1d_forward(const torch::Tensor& input, torch::Tensor& output, int k, int s, int p, int d);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("maxpool1d", &maxpool1d_forward, "Optimized Maxpool1D");
}
"""

fused_ext = load_inline(
    name='maxpool1d_optimized',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3'],
    with_cuda=True
)

def functional_model(x, *, maxpool_kernel_size, maxpool_stride, maxpool_padding, maxpool_dilation, maxpool_ceil_mode, maxpool_return_indices):
    total_len = x.size(2)
    # Correct floor/ceil output dimension calculation
    if maxpool_ceil_mode:
        output_length = ((total_len + 2 * maxpool_padding - maxpool_dilation * (maxpool_kernel_size - 1) - 1) + maxpool_stride - 1) // maxpool_stride + 1
    else:
        output_length = (total_len + 2 * maxpool_padding - maxpool_dilation * (maxpool_kernel_size - 1) - 1) // maxpool_stride + 1
    
    x_gpu = x.contiguous().cuda()
    output = torch.empty((x.size(0), x.size(1), output_length), device='cuda', dtype=x.dtype)
    
    fused_ext.maxpool1d(x_gpu, output, maxpool_kernel_size, maxpool_stride, maxpool_padding, maxpool_dilation)
    
    return output
