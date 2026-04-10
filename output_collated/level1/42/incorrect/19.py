# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_062618/code_10.py
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

# The CUDA kernel using coalesced memory access to improve performance
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16
#define MULT_FACTOR 2
#define OUTPUT_TILE (TILE_SIZE * MULT_FACTOR)

__global__ void max_pool2d_kernel_coalesced(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size, int channels,
    int in_h, int in_w,
    int out_h, int out_w,
    int k_size, int stride, int padding) {

    extern __shared__ float shared_input[];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;
    int total_threads = blockDim.x * blockDim.y;
    
    // Calculate which batch/channel we're processing
    int batch_idx = blockIdx.z / channels;
    int channel_idx = blockIdx.z % channels;
    
    // Input/output base pointers for this batch and channel
    const float* input_base = input + (batch_idx * channels + channel_idx) * in_h * in_w;
    float* output_base = output + (batch_idx * channels + channel_idx) * out_h * out_w;
    
    // Tile boundaries in output space
    int out_x_start = blockIdx.x * OUTPUT_TILE;
    int out_y_start = blockIdx.y * OUTPUT_TILE;
    
    // Corresponding input region boundaries
    int in_x_start = out_x_start * stride - padding;
    int in_y_start = out_y_start * stride - padding;
    int in_tile_width = OUTPUT_TILE * stride + k_size - 1;
    int in_tile_height = OUTPUT_TILE * stride + k_size - 1;
    
    // Shared memory dimensions
    const int shared_width = in_tile_width;
    const int shared_height = in_tile_height;
    const int shared_elements = shared_width * shared_height;
    
    // Coalesced loading of shared memory
    for (int i = tid; i < shared_elements; i += total_threads) {
        int sy = i / shared_width;
        int sx = i % shared_width;
        
        int ix = in_x_start + sx;
        int iy = in_y_start + sy;
        
        float value = -1e38f;
        if (ix >= 0 && ix < in_w && iy >= 0 && iy < in_h) {
            value = input_base[iy * in_w + ix];
        }
        shared_input[i] = value;
    }
    
    __syncthreads();
    
    // Compute output tiles
    #pragma unroll
    for (int sub_y = 0; sub_y < MULT_FACTOR; sub_y++) {
        #pragma unroll
        for (int sub_x = 0; sub_x < MULT_FACTOR; sub_x++) {
            int out_x = out_x_start + sub_x * TILE_SIZE + tx;
            int out_y = out_y_start + sub_y * TILE_SIZE + ty;
            
            if (out_x < out_w && out_y < out_h) {
                // Starting position in shared memory for this pooling window
                int shared_x_start = sub_x * TILE_SIZE * stride;
                int shared_y_start = sub_y * TILE_SIZE * stride;
                
                float max_val = -1e38f;
                
                // Unroll for small kernel sizes to maximize performance
                if(k_size == 2) {
                    float val1 = shared_input[(shared_y_start + 0) * shared_width + shared_x_start + 0];
                    float val2 = shared_input[(shared_y_start + 0) * shared_width + shared_x_start + 1];
                    float val3 = shared_input[(shared_y_start + 1) * shared_width + shared_x_start + 0];
                    float val4 = shared_input[(shared_y_start + 1) * shared_width + shared_x_start + 1];
                    if(val1 > max_val) max_val = val1;
                    if(val2 > max_val) max_val = val2;
                    if(val3 > max_val) max_val = val3;
                    if(val4 > max_val) max_val = val4;
                } else {
                    #pragma unroll
                    for (int ky = 0; ky < k_size; ky++) {
                        #pragma unroll
                        for (int kx = 0; kx < k_size; kx++) {
                            float val = shared_input[(shared_y_start + ky) * shared_width + shared_x_start + kx];
                            if (val > max_val) max_val = val;
                        }
                    }
                }
                
                output_base[out_y * out_w + out_x] = max_val;
            }
        }
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

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((out_w + OUTPUT_TILE - 1) / OUTPUT_TILE,
              (out_h + OUTPUT_TILE - 1) / OUTPUT_TILE,
              batch * channels);

    const int shared_width = OUTPUT_TILE * stride + k_size - 1;
    const int shared_height = OUTPUT_TILE * stride + k_size - 1;
    size_t shared_mem_size = shared_width * shared_height * sizeof(float);

    max_pool2d_kernel_coalesced<<<grid, block, shared_mem_size>>>(
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
    m.def("forward", &max_pool2d_forward, "Coalesced Max Pool 2D Forward");
}
"""

module = load_inline(name='max_pool2d_coalesced', cpp_sources=cpp_source, cuda_sources=cuda_kernel, 
                     extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True)

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
