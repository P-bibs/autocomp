# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_063920/code_11.py
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

# Optimized CUDA kernel with focus on occupancy and performance for RTX 2080Ti
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

__global__ void max_pool2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size, int channels,
    int in_h, int in_w,
    int out_h, int out_w,
    int k_size, int stride, int padding) {

    extern __shared__ float shared_input[];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Compute batch and channel from linear block index
    int block_id = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
    if (block_id >= batch_size * channels) return;
    
    int ch = block_id % channels;
    int b = block_id / channels;
    int bz = b * channels + ch;
    
    // Shared memory dimensions - optimized for occupancy
    const int shared_h = TILE_SIZE * stride + k_size - 1;
    const int shared_w = TILE_SIZE * stride + k_size - 1;
    const int total_threads = TILE_SIZE * TILE_SIZE;
    const int thread_id = ty * TILE_SIZE + tx;

    // Calculate start position for this tile
    int ih_start = (blockIdx.x % ((out_w + TILE_SIZE - 1) / TILE_SIZE)) * TILE_SIZE * stride - padding;
    int iw_start = (blockIdx.x % ((out_w + TILE_SIZE - 1) / TILE_SIZE)) * TILE_SIZE * stride - padding;
    
    // Adjust for correct tile positioning
    int tile_idx_x = blockIdx.x % ((out_w + TILE_SIZE - 1) / TILE_SIZE);
    int tile_idx_y = blockIdx.y % ((out_h + TILE_SIZE - 1) / TILE_SIZE);
    ih_start = tile_idx_y * TILE_SIZE * stride - padding;
    iw_start = tile_idx_x * TILE_SIZE * stride - padding;

    // Load into shared memory with coalesced access
    for (int i = thread_id; i < shared_h * shared_w; i += total_threads) {
        int r = i / shared_w;
        int c = i % shared_w;
        int ih = ih_start + r;
        int iw = iw_start + c;
        
        if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
            shared_input[i] = input[((bz * in_h + ih) * in_w + iw)];
        } else {
            shared_input[i] = -FLT_MAX;
        }
    }
    
    __syncthreads();
    
    // Compute one output pixel per thread
    int oh = tile_idx_y * TILE_SIZE + ty;
    int ow = tile_idx_x * TILE_SIZE + tx;
    
    if (oh < out_h && ow < out_w) {
        float max_val = -FLT_MAX;
        int start_r = ty * stride;
        int start_c = tx * stride;
        
        for (int i = 0; i < k_size; ++i) {
            for (int j = 0; j < k_size; ++j) {
                float val = shared_input[(start_r + i) * shared_w + (start_c + j)];
                max_val = fmaxf(max_val, val);
            }
        }
        output[((bz * out_h + oh) * out_w + ow)] = max_val;
    }
}

void max_pool2d_forward(const torch::Tensor& input, torch::Tensor& output, 
                        int k_size, int stride, int padding) {
    const int batch = input.size(0);
    const int channels = input.size(1);
    const int in_h = input.size(2);
    const int in_w = input.size(3);
    const int out_h = output.size(2);
    const int out_w = output.size(3);

    // Calculate grid dimensions
    int grid_x = (out_w + TILE_SIZE - 1) / TILE_SIZE;
    int grid_y = (out_h + TILE_SIZE - 1) / TILE_SIZE;
    int grid_z = batch * channels;
    
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid(grid_x, grid_y, 1);
    // Flatten grid to 1D for better scheduling
    int total_blocks = grid_x * grid_y * ((grid_z + grid_x * grid_y - 1) / (grid_x * grid_y));
    grid = dim3(min(total_blocks, 65535), 
                min((total_blocks + 65534) / 65535, 65535), 
                (total_blocks + 65534 * 65535) / (65535 * 65535));

    const int shared_h = TILE_SIZE * stride + k_size - 1;
    const int shared_w = TILE_SIZE * stride + k_size - 1;
    size_t shared_mem_size = shared_h * shared_w * sizeof(float);

    max_pool2d_kernel<<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        batch, channels, in_h, in_w, out_h, out_w,
        k_size, stride, padding
    );
}
"""

# Corrected version with proper grid/block indexing
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>

#define TILE_SIZE 32

__global__ void max_pool2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size, int channels,
    int in_h, int in_w,
    int out_h, int out_w,
    int k_size, int stride, int padding) {

    extern __shared__ float shared_input[];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Compute batch and channel from linear block ID
    int block_id = blockIdx.x;
    int ch = block_id % channels;
    int b = block_id / channels;
    
    if (b >= batch_size) return;
    int bz = b * channels + ch;
    
    // Shared memory dimensions
    const int shared_h = TILE_SIZE * stride + k_size - 1;
    const int shared_w = TILE_SIZE * stride + k_size - 1;
    const int total_threads = TILE_SIZE * TILE_SIZE;
    const int thread_id = ty * TILE_SIZE + tx;

    // Calculate start position for this tile
    int tile_idx_x = blockIdx.y % ((out_w + TILE_SIZE - 1) / TILE_SIZE);
    int tile_idx_y = blockIdx.y / ((out_w + TILE_SIZE - 1) / TILE_SIZE);
    
    int ih_start = tile_idx_y * TILE_SIZE * stride - padding;
    int iw_start = tile_idx_x * TILE_SIZE * stride - padding;

    // Load into shared memory with coalesced access
    for (int i = thread_id; i < shared_h * shared_w; i += total_threads) {
        int r = i / shared_w;
        int c = i % shared_w;
        int ih = ih_start + r;
        int iw = iw_start + c;
        
        if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
            shared_input[i] = input[((bz * in_h + ih) * in_w + iw)];
        } else {
            shared_input[i] = -FLT_MAX;
        }
    }
    
    __syncthreads();
    
    // Compute one output pixel per thread
    int oh = tile_idx_y * TILE_SIZE + ty;
    int ow = tile_idx_x * TILE_SIZE + tx;
    
    if (oh < out_h && ow < out_w) {
        float max_val = -FLT_MAX;
        int start_r = ty * stride;
        int start_c = tx * stride;
        
        for (int i = 0; i < k_size; ++i) {
            for (int j = 0; j < k_size; ++j) {
                float val = shared_input[(start_r + i) * shared_w + (start_c + j)];
                max_val = fmaxf(max_val, val);
            }
        }
        output[((bz * out_h + oh) * out_w + ow)] = max_val;
    }
}

void max_pool2d_forward(const torch::Tensor& input, torch::Tensor& output, 
                        int k_size, int stride, int padding) {
    const int batch = input.size(0);
    const int channels = input.size(1);
    const int in_h = input.size(2);
    const int in_w = input.size(3);
    const int out_h = output.size(2);
    const int out_w = output.size(3);

    // Calculate grid dimensions
    int grid_x = batch * channels;
    int grid_y = ((out_w + TILE_SIZE - 1) / TILE_SIZE) * ((out_h + TILE_SIZE - 1) / TILE_SIZE);
    
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid(grid_x, grid_y);

    const int shared_h = TILE_SIZE * stride + k_size - 1;
    const int shared_w = TILE_SIZE * stride + k_size - 1;
    size_t shared_mem_size = shared_h * shared_w * sizeof(float);

    max_pool2d_kernel<<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        batch, channels, in_h, in_w, out_h, out_w,
        k_size, stride, padding
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void max_pool2d_forward(const torch::Tensor& input, torch::Tensor& output, int k_size, int stride, int padding);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_forward, "Optimized MaxPool2D with improved occupancy");
}
"""

module = load_inline(
    name='max_pool2d_optimized',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, maxpool_kernel_size, maxpool_stride, maxpool_padding, maxpool_dilation, maxpool_ceil_mode, maxpool_return_indices):
    if maxpool_dilation != 1 or maxpool_ceil_mode or maxpool_return_indices:
        raise NotImplementedError("Only basic MaxPool2d supported.")
        
    h_in, w_in = x.shape[2], x.shape[3]
    h_out = (h_in + 2 * maxpool_padding - maxpool_kernel_size) // maxpool_stride + 1
    w_out = (w_in + 2 * maxpool_padding - maxpool_kernel_size) // maxpool_stride + 1
    
    output = torch.empty((x.shape[0], x.shape[1], h_out, w_out), device=x.device, dtype=x.dtype)
    module.forward(x.contiguous(), output, maxpool_kernel_size, maxpool_stride, maxpool_padding)
    return output
