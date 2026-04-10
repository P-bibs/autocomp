# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_053509/code_26.py
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

# --- CUDA Kernel with Shared Memory Tiling ---
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <algorithm>

__global__ void max_pool2d_kernel_tiled(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size, int channels,
    int in_h, int in_w,
    int out_h, int out_w,
    int k_size, int stride, int padding) {

    const int TILE_SIZE = 16;
    // Shared memory size for a block: (TILE_SIZE*stride + k_size) is safely bounded by a constant
    // 32*32 is sufficient for k_size up to 5 and stride up to 2
    __shared__ float s_data[32 * 32];

    int b = blockIdx.z / channels;
    int c = blockIdx.z % channels;
    int out_y = blockIdx.y * TILE_SIZE + threadIdx.y;
    int out_x = blockIdx.x * TILE_SIZE + threadIdx.x;

    int in_y_start = blockIdx.y * TILE_SIZE * stride - padding;
    int in_x_start = blockIdx.x * TILE_SIZE * stride - padding;

    // Load tile into shared memory (cooperatively)
    int tile_h = TILE_SIZE * stride + k_size - 1;
    int tile_w = TILE_SIZE * stride + k_size - 1;

    for (int i = threadIdx.y; i < tile_h; i += TILE_SIZE) {
        for (int j = threadIdx.x; j < tile_w; j += TILE_SIZE) {
            int cur_y = in_y_start + i;
            int cur_x = in_x_start + j;
            if (cur_y >= 0 && cur_y < in_h && cur_x >= 0 && cur_x < in_w) {
                s_data[i * tile_w + j] = input[((b * channels + c) * in_h + cur_y) * in_w + cur_x];
            } else {
                s_data[i * tile_w + j] = -1e38f;
            }
        }
    }
    __syncthreads();

    if (out_y < out_h && out_x < out_w) {
        float max_val = -1e38f;
        int start_y = (out_y * stride - padding) - in_y_start;
        int start_x = (out_x * stride - padding) - in_x_start;

        for (int i = 0; i < k_size; ++i) {
            for (int j = 0; j < k_size; ++j) {
                max_val = fmaxf(max_val, s_data[(start_y + i) * tile_w + (start_x + j)]);
            }
        }
        output[((b * channels + c) * out_h + out_y) * out_w + out_x] = max_val;
    }
}

void launch_max_pool2d(const torch::Tensor& input, torch::Tensor& output, 
                       int k_size, int stride, int padding) {
    int b = input.size(0);
    int c = input.size(1);
    int in_h = input.size(2);
    int in_w = input.size(3);
    int out_h = output.size(2);
    int out_w = output.size(3);

    dim3 threads(16, 16);
    dim3 blocks((out_w + 15) / 16, (out_h + 15) / 16, b * c);

    max_pool2d_kernel_tiled<<<blocks, threads>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        b, c, in_h, in_w, out_h, out_w, k_size, stride, padding
    );
}
"""

# --- C++ Binding ---
cpp_source = r"""
#include <torch/extension.h>
void launch_max_pool2d(const torch::Tensor& input, torch::Tensor& output, int k_size, int stride, int padding);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &launch_max_pool2d, "Tiled Max Pool 2D");
}
"""

module = load_inline(
    name='max_pool2d_tiled',
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
