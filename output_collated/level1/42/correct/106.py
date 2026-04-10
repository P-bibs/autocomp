# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_062618/code_27.py
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

# The CUDA kernel with optimized loop unrolling, register-level cache, and index hoisting
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
    int bz = blockIdx.z;
    
    const int MULT_FACTOR = 2;
    const int OUTPUT_TILE = TILE_SIZE * MULT_FACTOR;
    const int shared_dim = OUTPUT_TILE * stride + k_size - 1;
    const int total_threads = TILE_SIZE * TILE_SIZE;
    const int tile_elements = shared_dim * shared_dim;

    int ih_start = blockIdx.y * OUTPUT_TILE * stride - padding;
    int iw_start = blockIdx.x * OUTPUT_TILE * stride - padding;

    // Coalesced loads into shared memory
    for (int i = ty * TILE_SIZE + tx; i < tile_elements; i += total_threads) {
        int ih = ih_start + i / shared_dim;
        int iw = iw_start + i % shared_dim;
        bool valid = (ih >= 0) & (ih < in_h) & (iw >= 0) & (iw < in_w);
        shared_input[i] = valid ? input[(bz * in_h + ih) * in_w + iw] : -1e38f;
    }
    
    __syncthreads();
    
    for (int sub_y = 0; sub_y < MULT_FACTOR; sub_y++) {
        for (int sub_x = 0; sub_x < MULT_FACTOR; sub_x++) {
            int ow = blockIdx.x * OUTPUT_TILE + sub_x * TILE_SIZE + tx;
            int oh = blockIdx.y * OUTPUT_TILE + sub_y * TILE_SIZE + ty;
            
            if (ow < out_w && oh < out_h) {
                float max_val = -1e38f;
                int start_row = (sub_y * TILE_SIZE + ty) * stride;
                int start_col = (sub_x * TILE_SIZE + tx) * stride;
                
                if (k_size == 2) {
                    int r0 = start_row * shared_dim + start_col;
                    int r1 = (start_row + 1) * shared_dim + start_col;
                    max_val = fmaxf(fmaxf(shared_input[r0], shared_input[r0 + 1]), 
                                    fmaxf(shared_input[r1], shared_input[r1 + 1]));
                } else {
                    #pragma unroll
                    for (int i = 0; i < k_size; ++i) {
                        int row_offset = (start_row + i) * shared_dim + start_col;
                        #pragma unroll
                        for (int j = 0; j < k_size; ++j) {
                            max_val = fmaxf(max_val, shared_input[row_offset + j]);
                        }
                    }
                }
                output[((bz * out_h + oh) * out_w + ow)] = max_val;
            }
        }
    }
}

void max_pool_forward(const torch::Tensor& input, torch::Tensor& output, 
                     int k_size, int stride, int padding) {
    int batch = input.size(0);
    int channels = input.size(1);
    int in_h = input.size(2);
    int in_w = input.size(3);
    int out_h = output.size(2);
    int out_w = output.size(3);

    const int OUTPUT_TILE = TILE_SIZE * 2;
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((out_w + OUTPUT_TILE - 1) / OUTPUT_TILE,
              (out_h + OUTPUT_TILE - 1) / OUTPUT_TILE,
              batch * channels);

    size_t shared_mem_size = (OUTPUT_TILE * stride + k_size - 1) * (OUTPUT_TILE * stride + k_size - 1) * sizeof(float);

    max_pool2d_kernel<<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        batch, channels, in_h, in_w, out_h, out_w,
        k_size, stride, padding
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void max_pool_forward(const torch::Tensor& input, torch::Tensor& output, int k_size, int stride, int padding);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool_forward, "Optimized MaxPool2D");
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
    h_in, w_in = x.shape[2], x.shape[3]
    h_out = (h_in + 2 * maxpool_padding - maxpool_kernel_size) // maxpool_stride + 1
    w_out = (w_in + 2 * maxpool_padding - maxpool_kernel_size) // maxpool_stride + 1
    
    output = torch.empty((x.shape[0], x.shape[1], h_out, w_out), device=x.device, dtype=x.dtype)
    module.forward(x.contiguous(), output, maxpool_kernel_size, maxpool_stride, maxpool_padding)
    return output
