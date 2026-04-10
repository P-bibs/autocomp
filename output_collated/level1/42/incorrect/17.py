# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_062618/code_3.py
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

# The CUDA kernel with enhanced tiling for optimal cache utilization
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

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
    int tid = ty * TILE_SIZE + tx;
    
    // Optimize tile size based on kernel size for better cache utilization
    const int TILE_EXPAND = (k_size > 2) ? 1 : 0;
    const int EFFECTIVE_TILE = TILE_SIZE + TILE_EXPAND;
    const int OUTPUT_TILE = EFFECTIVE_TILE;
    const int shared_dim = OUTPUT_TILE * stride + k_size - 1;
    const int total_threads = TILE_SIZE * TILE_SIZE;
    const int tile_elements = shared_dim * shared_dim;

    // Calculate block indices for batch and channel dimensions
    int batch_idx = blockIdx.z / channels;
    int channel_idx = blockIdx.z % channels;
    
    // Starting global input coordinates for this tile
    int ih_start = blockIdx.y * OUTPUT_TILE * stride - padding;
    int iw_start = blockIdx.x * OUTPUT_TILE * stride - padding;

    // Coalesced loading into shared memory with boundary checking
    for (int i = tid; i < tile_elements; i += total_threads) {
        int row = i / shared_dim;
        int col = i % shared_dim;
        int ih = ih_start + row;
        int iw = iw_start + col;
        
        bool valid = (ih >= 0) && (ih < in_h) && (iw >= 0) && (iw < in_w);
        int input_idx = ((batch_idx * channels + channel_idx) * in_h + ih) * in_w + iw;
        shared_input[i] = valid ? input[input_idx] : -1e38f;
    }
    
    __syncthreads();
    
    // Each thread computes one output point
    int ow = blockIdx.x * OUTPUT_TILE + tx;
    int oh = blockIdx.y * OUTPUT_TILE + ty;
    
    if (ow < out_w && oh < out_h) {
        float max_val = -1e38f;
        int start_row = ty * stride;
        int start_col = tx * stride;
        
        // Optimized pooling based on kernel size
        if (k_size == 2) {
            // Fully unrolled for k=2 case - optimal cache utilization
            int base_idx1 = start_row * shared_dim + start_col;
            int base_idx2 = (start_row + 1) * shared_dim + start_col;
            
            float val1 = shared_input[base_idx1];
            float val2 = shared_input[base_idx1 + 1];
            float val3 = shared_input[base_idx2];
            float val4 = shared_input[base_idx2 + 1];
            
            max_val = fmaxf(fmaxf(fmaxf(val1, val2), val3), val4);
        } else if (k_size == 3) {
            // Fully unrolled for k=3 case - better cache hit rate
            int base_idx1 = start_row * shared_dim + start_col;
            int base_idx2 = (start_row + 1) * shared_dim + start_col;
            int base_idx3 = (start_row + 2) * shared_dim + start_col;
            
            float val1 = shared_input[base_idx1];
            float val2 = shared_input[base_idx1 + 1];
            float val3 = shared_input[base_idx1 + 2];
            float val4 = shared_input[base_idx2];
            float val5 = shared_input[base_idx2 + 1];
            float val6 = shared_input[base_idx2 + 2];
            float val7 = shared_input[base_idx3];
            float val8 = shared_input[base_idx3 + 1];
            float val9 = shared_input[base_idx3 + 2];
            
            max_val = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(val1, val2), val3), val4), val5), val6), val7), val8), val9);
        } else {
            // General case with loop unrolling for better cache utilization
            #pragma unroll 4
            for (int i = 0; i < k_size; ++i) {
                int row_offset = (start_row + i) * shared_dim + start_col;
                #pragma unroll 4
                for (int j = 0; j < k_size; ++j) {
                    float val = shared_input[row_offset + j];
                    max_val = fmaxf(max_val, val);
                }
            }
        }
        
        int output_idx = ((batch_idx * channels + channel_idx) * out_h + oh) * out_w + ow;
        output[output_idx] = max_val;
    }
}

void max_pool2d_forward(const torch::Tensor& input, torch::Tensor& output, 
                        int k_size, int stride, int padding) {
    int batch = input.size(0);
    int channels = input.size(1);
    int in_h = input.size(2);
    int in_w = input.size(3);
    int out_h = output.size(2);
    int out_w = output.size(3);

    // Optimize tile size based on kernel size
    const int TILE_EXPAND = (k_size > 2) ? 1 : 0;
    const int EFFECTIVE_TILE = TILE_SIZE + TILE_EXPAND;
    const int OUTPUT_TILE = EFFECTIVE_TILE;

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((out_w + OUTPUT_TILE - 1) / OUTPUT_TILE,
              (out_h + OUTPUT_TILE - 1) / OUTPUT_TILE,
              batch * channels);

    const int shared_dim = OUTPUT_TILE * stride + k_size - 1;
    size_t shared_mem_size = shared_dim * shared_dim * sizeof(float);

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
    m.def("forward", &max_pool2d_forward, "Max Pool 2D Forward with Optimized Tiling");
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
    if maxpool_dilation != 1:
        raise NotImplementedError("Dilation not supported in optimized kernel.")
    if maxpool_ceil_mode:
        raise NotImplementedError("Ceil mode not supported in optimized kernel.")
    if maxpool_return_indices:
        raise NotImplementedError("Indices not implemented.")
        
    h_in, w_in = x.shape[2], x.shape[3]
    h_out = (h_in + 2 * maxpool_padding - maxpool_kernel_size) // maxpool_stride + 1
    w_out = (w_in + 2 * maxpool_padding - maxpool_kernel_size) // maxpool_stride + 1
    
    output = torch.empty((x.shape[0], x.shape[1], h_out, w_out), device=x.device, dtype=x.dtype)
    module.forward(x.contiguous(), output, maxpool_kernel_size, maxpool_stride, maxpool_padding)
    return output
