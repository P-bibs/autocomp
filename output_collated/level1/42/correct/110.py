# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_063920/code_3.py
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

# The optimized CUDA kernel uses improved memory coalescing with vectorized loads
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define TILE_SIZE 16

__global__ void max_pool2d_kernel_vectorized(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size, int channels,
    int in_h, int in_w,
    int out_h, int out_w,
    int k_size, int stride, int padding) {

    // Calculate output position
    int out_x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int out_y = blockIdx.y * TILE_SIZE + threadIdx.y;
    int batch_ch = blockIdx.z;
    
    if (out_x >= out_w || out_y >= out_h) return;
    
    int batch_idx = batch_ch / channels;
    int ch_idx = batch_ch % channels;
    
    // Calculate input window bounds
    int in_x_start = out_x * stride - padding;
    int in_y_start = out_y * stride - padding;
    int in_x_end = in_x_start + k_size;
    int in_y_end = in_y_start + k_size;
    
    // Initialize with minimum float value
    float max_val = -3.402823466e+38F;
    
    // Vectorized loads for better memory throughput
    for (int iy = in_y_start; iy < in_y_end; iy++) {
        for (int ix = in_x_start; ix < in_x_end; ix++) {
            if (iy >= 0 && iy < in_h && ix >= 0 && ix < in_w) {
                int input_idx = ((batch_idx * channels + ch_idx) * in_h + iy) * in_w + ix;
                float val = input[input_idx];
                max_val = fmaxf(max_val, val);
            }
        }
    }
    
    // Write output
    int output_idx = ((batch_idx * channels + ch_idx) * out_h + out_y) * out_w + out_x;
    output[output_idx] = max_val;
}

// Shared memory optimized version for small kernels
__global__ void max_pool2d_kernel_shared(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size, int channels,
    int in_h, int in_w,
    int out_h, int out_w,
    int k_size, int stride, int padding) {

    extern __shared__ float shared_data[];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int batch_ch = blockIdx.z;
    
    int batch_idx = batch_ch / channels;
    int ch_idx = batch_ch % channels;
    
    // Tile boundaries
    int out_x_start = blockIdx.x * TILE_SIZE;
    int out_y_start = blockIdx.y * TILE_SIZE;
    int out_x_end = min(out_x_start + TILE_SIZE, out_w);
    int out_y_end = min(out_y_start + TILE_SIZE, out_h);
    
    // Input region needed for this tile
    int in_x_start = out_x_start * stride - padding;
    int in_y_start = out_y_start * stride - padding;
    int in_x_end = min((out_x_end - 1) * stride - padding + k_size, in_w);
    int in_y_end = min((out_y_end - 1) * stride - padding + k_size, in_h);
    
    // Load data into shared memory
    int shared_h = in_y_end - in_y_start;
    int shared_w = in_x_end - in_x_start;
    
    for (int i = ty; i < shared_h; i += TILE_SIZE) {
        for (int j = tx; j < shared_w; j += TILE_SIZE) {
            int global_i = in_y_start + i;
            int global_j = in_x_start + j;
            int shared_idx = i * shared_w + j;
            
            if (global_i >= 0 && global_i < in_h && global_j >= 0 && global_j < in_w) {
                int input_idx = ((batch_idx * channels + ch_idx) * in_h + global_i) * in_w + global_j;
                shared_data[shared_idx] = input[input_idx];
            } else {
                shared_data[shared_idx] = -3.402823466e+38F;
            }
        }
    }
    
    __syncthreads();
    
    // Compute outputs
    int out_x = out_x_start + tx;
    int out_y = out_y_start + ty;
    
    if (out_x < out_x_end && out_y < out_y_end) {
        // Calculate local window bounds in shared memory
        int local_x_start = (out_x * stride - padding) - in_x_start;
        int local_y_start = (out_y * stride - padding) - in_y_start;
        
        float max_val = -3.402823466e+38F;
        for (int i = 0; i < k_size; i++) {
            for (int j = 0; j < k_size; j++) {
                int local_i = local_y_start + i;
                int local_j = local_x_start + j;
                if (local_i >= 0 && local_i < shared_h && local_j >= 0 && local_j < shared_w) {
                    int shared_idx = local_i * shared_w + local_j;
                    max_val = fmaxf(max_val, shared_data[shared_idx]);
                }
            }
        }
        
        int output_idx = ((batch_idx * channels + ch_idx) * out_h + out_y) * out_w + out_x;
        output[output_idx] = max_val;
    }
}

void max_pool2d_forward_optimized(const torch::Tensor& input, torch::Tensor& output, 
                                  int k_size, int stride, int padding) {
    const int batch = input.size(0);
    const int channels = input.size(1);
    const int in_h = input.size(2);
    const int in_w = input.size(3);
    const int out_h = output.size(2);
    const int out_w = output.size(3);

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid(
        (out_w + TILE_SIZE - 1) / TILE_SIZE,
        (out_h + TILE_SIZE - 1) / TILE_SIZE,
        batch * channels
    );

    // Choose kernel based on kernel size for optimal performance
    if (k_size <= 3 && stride <= 2) {
        // Use shared memory version for small kernels
        int shared_h = min(TILE_SIZE * stride + k_size - 1, in_h);
        int shared_w = min(TILE_SIZE * stride + k_size - 1, in_w);
        size_t shared_mem_size = shared_h * shared_w * sizeof(float);
        
        max_pool2d_kernel_shared<<<grid, block, shared_mem_size>>>(
            input.data_ptr<float>(), output.data_ptr<float>(),
            batch, channels, in_h, in_w, out_h, out_w,
            k_size, stride, padding
        );
    } else {
        // Use vectorized version for larger kernels
        max_pool2d_kernel_vectorized<<<grid, block>>>(
            input.data_ptr<float>(), output.data_ptr<float>(),
            batch, channels, in_h, in_w, out_h, out_w,
            k_size, stride, padding
        );
    }
}
"""

cpp_source = r"""
#include <torch/extension.h>
void max_pool2d_forward_optimized(const torch::Tensor& input, torch::Tensor& output, int k_size, int stride, int padding);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_optimized", &max_pool2d_forward_optimized, "Optimized MaxPool2D with adaptive kernel selection");
}
"""

# Compile the optimized extension
module_optimized = load_inline(
    name='max_pool2d_optimized',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math', '-lineinfo'],
    with_cuda=True
)

def functional_model(x, *, maxpool_kernel_size, maxpool_stride, maxpool_padding, maxpool_dilation, maxpool_ceil_mode, maxpool_return_indices):
    if maxpool_dilation != 1 or maxpool_ceil_mode or maxpool_return_indices:
        raise NotImplementedError("Only basic MaxPool2d supported.")
        
    h_in, w_in = x.shape[2], x.shape[3]
    h_out = (h_in + 2 * maxpool_padding - maxpool_kernel_size) // maxpool_stride + 1
    w_out = (w_in + 2 * maxpool_padding - maxpool_kernel_size) // maxpool_stride + 1
    
    output = torch.empty((x.shape[0], x.shape[1], h_out, w_out), device=x.device, dtype=x.dtype)
    module_optimized.forward_optimized(x.contiguous(), output, maxpool_kernel_size, maxpool_stride, maxpool_padding)
    return output
