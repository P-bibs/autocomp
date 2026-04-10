# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_062618/code_17.py
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

# The CUDA kernel uses optimized shared memory tiling and vectorized-like thread logic
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>

#define TILE_SIZE 16

__global__ void max_pool2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size, int channels,
    int in_h, int in_w,
    int out_h, int out_w,
    int k_size, int stride, int padding) {

    extern __shared__ float s_data[];

    int b = blockIdx.z / channels;
    int c = blockIdx.z % channels;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int bc_offset = (b * channels + c);
    int in_c_offset = bc_offset * in_h * in_w;
    int out_c_offset = bc_offset * out_h * out_w;

    // Load tile into shared memory (Coalesced)
    int shared_dim = TILE_SIZE * stride + k_size - 1;
    int tile_origin_h = blockIdx.y * TILE_SIZE * stride - padding;
    int tile_origin_w = blockIdx.x * TILE_SIZE * stride - padding;

    for (int i = ty; i < shared_dim; i += blockDim.y) {
        for (int j = tx; j < shared_dim; j += blockDim.x) {
            int ih = tile_origin_h + i;
            int iw = tile_origin_w + j;
            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w)
                s_data[i * shared_dim + j] = input[in_c_offset + ih * in_w + iw];
            else
                s_data[i * shared_dim + j] = -FLT_MAX;
        }
    }
    __syncthreads();

    int oh = blockIdx.y * TILE_SIZE + ty;
    int ow = blockIdx.x * TILE_SIZE + tx;

    if (oh < out_h && ow < out_w) {
        float max_val = -FLT_MAX;
        int start_h = ty * stride;
        int start_w = tx * stride;
        for (int i = 0; i < k_size; ++i) {
            for (int j = 0; j < k_size; ++j) {
                float val = s_data[(start_h + i) * shared_dim + (start_w + j)];
                if (val > max_val) max_val = val;
            }
        }
        output[out_c_offset + oh * out_w + ow] = max_val;
    }
}

void max_pool2d_forward(const torch::Tensor& input, torch::Tensor& output, 
                        int k_size, int stride, int padding) {
    const int batch = input.size(0);
    const int channels = input.size(1);
    const int out_h = output.size(2);
    const int out_w = output.size(3);

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((out_w + TILE_SIZE - 1) / TILE_SIZE,
              (out_h + TILE_SIZE - 1) / TILE_SIZE,
              batch * channels);

    int shared_dim = TILE_SIZE * stride + k_size - 1;
    size_t shared_mem_size = shared_dim * shared_dim * sizeof(float);

    max_pool2d_kernel<<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        batch, channels, input.size(2), input.size(3), out_h, out_w,
        k_size, stride, padding
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void max_pool2d_forward(const torch::Tensor& input, torch::Tensor& output, int k_size, int stride, int padding);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_forward, "Optimized MaxPool2D Forward");
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
