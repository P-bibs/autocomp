# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_053509/code_28.py
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

# Optimized CUDA Kernel for MaxPool2d
# Plan: 
# 1. Coalesce global memory loads by letting threads fetch segments of the shared memory tile.
# 2. Use __ldg() for read-only access to global memory.
# 3. Use #pragma unroll to maximize throughput in the pooling window loops.
# 4. Minimize index arithmetic in inner loops.

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void max_pool2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int channels,
    const int in_h,
    const int in_w,
    const int out_h,
    const int out_w,
    const int k_size,
    const int stride,
    const int padding) {

    extern __shared__ float shared_input[];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int bz = blockIdx.z; 

    // Calculate dimensions
    const int shared_width = TILE_SIZE * stride + k_size - 1;
    const int shared_height = TILE_SIZE * stride + k_size - 1;
    const int shared_num_elements = shared_width * shared_height;
    
    // Global coordinate base
    const int input_offset = bz * in_h * in_w;
    const int iw_base = bx * TILE_SIZE * stride - padding;
    const int ih_base = by * TILE_SIZE * stride - padding;

    // Coalesced loading into shared memory
    const int tid = ty * TILE_SIZE + tx;
    const int num_threads = TILE_SIZE * TILE_SIZE;

    for (int i = tid; i < shared_num_elements; i += num_threads) {
        int r = i / shared_width;
        int c = i % shared_width;
        int ih = ih_base + r;
        int iw = iw_base + c;
        
        float val = -1e38f;
        if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
            val = __ldg(&input[input_offset + ih * in_w + iw]);
        }
        shared_input[i] = val;
    }

    __syncthreads();

    // Compute max for output tile
    int ow = bx * TILE_SIZE + tx;
    int oh = by * TILE_SIZE + ty;

    if (ow < out_w && oh < out_h) {
        float max_val = -1e38f;
        int start_r = ty * stride;
        int start_c = tx * stride;

        #pragma unroll
        for (int i = 0; i < k_size; ++i) {
            #pragma unroll
            for (int j = 0; j < k_size; ++j) {
                float val = shared_input[(start_r + i) * shared_width + (start_c + j)];
                if (val > max_val) max_val = val;
            }
        }
        output[bz * out_h * out_w + oh * out_w + ow] = max_val;
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
    dim3 grid((out_w + TILE_SIZE - 1) / TILE_SIZE, 
              (out_h + TILE_SIZE - 1) / TILE_SIZE, 
              batch * channels);

    int shared_width = TILE_SIZE * stride + k_size - 1;
    int shared_height = TILE_SIZE * stride + k_size - 1;
    size_t shared_size = shared_width * shared_height * sizeof(float);

    max_pool2d_kernel<<<grid, block, shared_size>>>(
        input.contiguous().data_ptr<float>(),
        output.data_ptr<float>(),
        channels, in_h, in_w, out_h, out_w, k_size, stride, padding
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void max_pool2d_forward(const torch::Tensor& input, torch::Tensor& output, int k_size, int stride, int padding);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_forward, "Max Pool 2D Forward");
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
    if maxpool_return_indices:
        raise NotImplementedError("Indices not supported.")
    
    h_in, w_in = x.shape[2], x.shape[3]
    h_out = (h_in + 2 * maxpool_padding - maxpool_kernel_size) // maxpool_stride + 1
    w_out = (w_in + 2 * maxpool_padding - maxpool_kernel_size) // maxpool_stride + 1
    
    output = torch.empty((x.shape[0], x.shape[1], h_out, w_out), device=x.device, dtype=x.dtype)
    module.forward(x, output, maxpool_kernel_size, maxpool_stride, maxpool_padding)
    return output
