# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_061245/code_21.py
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

# Defining vectorized max pooling CUDA kernel with shared memory optimization
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// TILE_DIM 16x16 threads
#define TILE_DIM 16

__global__ void max_pool2d_kernel(
    const float* __restrict__ input, 
    float* __restrict__ output,
    int in_h, int in_w, 
    int out_h, int out_w,
    int k_size, int stride, int padding) {

    // Shared memory for a tile of the input
    // Calculate dimensions required for the shared memory buffer
    // Adding 1 to height to prevent bank conflicts
    extern __shared__ float s_data[];

    int b_idx = blockIdx.z;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int tile_h = TILE_DIM * stride + k_size - 1;
    int tile_w = TILE_DIM * stride + k_size - 1;
    
    // Width with padding to avoid shared memory bank conflicts
    int shared_stride = tile_w + 1;

    int start_y = blockIdx.y * TILE_DIM * stride - padding;
    int start_x = blockIdx.x * TILE_DIM * stride - padding;

    // Cooperative loading of input tile into shared memory
    // Each thread loads elements to cover the needed region
    for (int i = ty; i < tile_h; i += TILE_DIM) {
        for (int j = tx; j < tile_w; j += TILE_DIM) {
            int ih = start_y + i;
            int iw = start_x + j;
            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                s_data[i * shared_stride + j] = input[b_idx * in_h * in_w + ih * in_w + iw];
            } else {
                s_data[i * shared_stride + j] = -1e38f; // Represent -inf
            }
        }
    }
    __syncthreads();

    // Perform max pooling
    int oh = blockIdx.y * TILE_DIM + ty;
    int ow = blockIdx.x * TILE_DIM + tx;

    if (oh < out_h && ow < out_w) {
        float max_val = -1e38f;
        int s_y = ty * stride;
        int s_x = tx * stride;
        
        #pragma unroll
        for (int i = 0; i < k_size; ++i) {
            #pragma unroll
            for (int j = 0; j < k_size; ++j) {
                float val = s_data[(s_y + i) * shared_stride + (s_x + j)];
                if (val > max_val) max_val = val;
            }
        }
        output[b_idx * out_h * out_w + oh * out_w + ow] = max_val;
    }
}

void max_pool2d_forward_cuda(const torch::Tensor& input, torch::Tensor& output, 
                             int k_size, int stride, int padding) {
    int batch = input.size(0);
    int channels = input.size(1);
    int in_h = input.size(2);
    int in_w = input.size(3);
    int out_h = output.size(2);
    int out_w = output.size(3);

    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((out_w + TILE_DIM - 1) / TILE_DIM, (out_h + TILE_DIM - 1) / TILE_DIM, batch * channels);

    int tile_h = TILE_DIM * stride + k_size - 1;
    int tile_w = TILE_DIM * stride + k_size - 1;
    size_t shared_mem_size = (tile_h * (tile_w + 1)) * sizeof(float);

    max_pool2d_kernel<<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        in_h, in_w, out_h, out_w, k_size, stride, padding
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void max_pool2d_forward_cuda(const torch::Tensor& input, torch::Tensor& output, int k_size, int stride, int padding);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_forward_cuda, "Max Pool 2D Forward");
}
"""

module = load_inline(name='max_pool2d_optimized', cpp_sources=cpp_source, cuda_sources=cuda_kernel, 
                     extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True)

def functional_model(x, *, maxpool_kernel_size, maxpool_stride, maxpool_padding, maxpool_dilation=1, maxpool_ceil_mode=False, maxpool_return_indices=False):
    if maxpool_dilation != 1 or maxpool_ceil_mode or maxpool_return_indices:
        raise NotImplementedError("Only standard max pooling is supported.")
        
    h_in, w_in = x.shape[2], x.shape[3]
    h_out = (h_in + 2 * maxpool_padding - maxpool_kernel_size) // maxpool_stride + 1
    w_out = (w_in + 2 * maxpool_padding - maxpool_kernel_size) // maxpool_stride + 1
    
    output = torch.empty((x.shape[0], x.shape[1], h_out, w_out), device=x.device, dtype=x.dtype)
    module.forward(x.contiguous(), output, maxpool_kernel_size, maxpool_stride, maxpool_padding)
    return output
