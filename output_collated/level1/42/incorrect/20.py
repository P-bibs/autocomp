# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_062618/code_9.py
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

# The CUDA kernel uses vectorized float4 loads for higher memory bandwidth utilization
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16
#define MULT_FACTOR 2

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
    int bz = blockIdx.z; // batch_size * channels
    
    const int OUTPUT_TILE = TILE_SIZE * MULT_FACTOR;
    const int shared_dim = OUTPUT_TILE * stride + k_size - 1;
    // Align to 4 elements for vectorization
    const int shared_dim_aligned = ((shared_dim + 3) / 4) * 4;
    const int total_threads = TILE_SIZE * TILE_SIZE;
    const int thread_id = ty * TILE_SIZE + tx;
    const int tile_elements = shared_dim * shared_dim_aligned;

    // Calculate start position for this tile
    int ih_start = blockIdx.y * OUTPUT_TILE * stride - padding;
    int iw_start = blockIdx.x * OUTPUT_TILE * stride - padding;

    // Load into shared memory with vectorized float4 access where possible
    const float4* input_float4 = (const float4*)input;
    float* shared_input_ptr = shared_input;
    
    // Vectorized loading: process 4 elements at a time
    for (int i = thread_id * 4; i < tile_elements; i += total_threads * 4) {
        int linear_idx = i;
        int r = linear_idx / shared_dim_aligned;
        int c = linear_idx % shared_dim_aligned;
        
        // Check if we can do a vectorized load (4 consecutive elements in a row)
        if (c + 4 <= shared_dim_aligned && r < shared_dim) {
            int ih = ih_start + r;
            
            // Vectorized load attempt - load 4 consecutive width elements
            if (ih >= 0 && ih < in_h) {
                int iw_base = iw_start + c;
                
                // Check if all 4 elements are within bounds
                if (iw_base >= 0 && iw_base + 3 < in_w) {
                    // Fully in-bounds: use vectorized float4 load
                    int input_offset = ((bz * in_h + ih) * in_w + iw_base) / 4;
                    float4 data = input_float4[input_offset];
                    
                    *((float4*)(&shared_input_ptr[linear_idx])) = data;
                } else {
                    // Partially out of bounds: load individually
                    for (int j = 0; j < 4; j++) {
                        int iw = iw_base + j;
                        if (iw >= 0 && iw < in_w) {
                            shared_input_ptr[linear_idx + j] = input[((bz * in_h + ih) * in_w + iw)];
                        } else {
                            shared_input_ptr[linear_idx + j] = -3.402823466e+38F;
                        }
                    }
                }
            } else {
                // Row out of bounds
                for (int j = 0; j < 4; j++) {
                    shared_input_ptr[linear_idx + j] = -3.402823466e+38F;
                }
            }
        } else {
            // Handle remaining elements that don't align to 4-element boundaries
            for (int k = 0; k < 4; k++) {
                int idx = linear_idx + k;
                if (idx >= tile_elements) break;
                
                int r = idx / shared_dim_aligned;
                int c = idx % shared_dim_aligned;
                
                if (c < shared_dim && r < shared_dim) {
                    int ih = ih_start + r;
                    int iw = iw_start + c;
                    
                    if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                        shared_input_ptr[idx] = input[((bz * in_h + ih) * in_w + iw)];
                    } else {
                        shared_input_ptr[idx] = -3.402823466e+38F;
                    }
                }
            }
        }
    }
    
    __syncthreads();
    
    // Each thread calculates a 2x2 patch of output pixels
    #pragma unroll
    for (int sub_y = 0; sub_y < MULT_FACTOR; sub_y++) {
        #pragma unroll
        for (int sub_x = 0; sub_x < MULT_FACTOR; sub_x++) {
            int oh = blockIdx.y * OUTPUT_TILE + sub_y * TILE_SIZE + ty;
            int ow = blockIdx.x * OUTPUT_TILE + sub_x * TILE_SIZE + tx;
            
            if (oh < out_h && ow < out_w) {
                float max_val = -3.402823466e+38F;
                int start_r = (sub_y * TILE_SIZE + ty) * stride;
                int start_c = (sub_x * TILE_SIZE + tx) * stride;
                
                #pragma unroll
                for (int i = 0; i < k_size; ++i) {
                    #pragma unroll
                    for (int j = 0; j < k_size; ++j) {
                        float val = shared_input[(start_r + i) * shared_dim_aligned + (start_c + j)];
                        if (val > max_val) max_val = val;
                    }
                }
                output[((bz * out_h + oh) * out_w + ow)] = max_val;
            }
        }
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

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((out_w + (TILE_SIZE * MULT_FACTOR) - 1) / (TILE_SIZE * MULT_FACTOR),
              (out_h + (TILE_SIZE * MULT_FACTOR) - 1) / (TILE_SIZE * MULT_FACTOR),
              batch * channels);

    const int shared_dim = (TILE_SIZE * MULT_FACTOR) * stride + k_size - 1;
    const int shared_dim_aligned = ((shared_dim + 3) / 4) * 4;
    size_t shared_mem_size = shared_dim * shared_dim_aligned * sizeof(float);

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
    m.def("forward", &max_pool2d_forward, "Vectorized MaxPool2D using float4");
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
