# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_063920/code_29.py
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

# The CUDA kernel implements a tiled, shared-memory MaxPool2D operation.
# Performance optimizations include:
# 1. Hoisting address arithmetic: Base shared memory indices are precalculated to reduce integer ALUs.
# 2. Shared Memory Tiling: Reduced global memory round-trips via coalesced block-level loads.
# 3. Unrolling: Fully unrolled inner loops to eliminate branching overhead.
# 4. Fast Math & __ldg: Leveraging hardware-specific caching for read-only input buffers.

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>

#define TILE_H 16
#define TILE_W 16
#define MULT_Y 4
#define MULT_X 4

__global__ void max_pool2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size, int channels,
    int in_h, int in_w,
    int out_h, int out_w,
    int k_size, int stride, int padding) {

    extern __shared__ float shared_input[];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bz = blockIdx.z;

    const int out_x = blockIdx.x * (TILE_W * MULT_X) + tx;
    const int out_y = blockIdx.y * (TILE_H * MULT_Y) + ty;
    
    // Calculate shared memory dimensions once per block
    const int s_w = (TILE_W * MULT_X) * stride + k_size - 1;
    const int s_h = (TILE_H * MULT_Y) * stride + k_size - 1;
    
    const int start_x = blockIdx.x * (TILE_W * MULT_X) * stride - padding;
    const int start_y = blockIdx.y * (TILE_H * MULT_Y) * stride - padding;

    // Collaborative loading into shared memory
    for (int i = ty; i < s_h; i += blockDim.y) {
        for (int j = tx; j < s_w; j += blockDim.x) {
            int ih = start_y + i;
            int iw = start_x + j;
            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                shared_input[i * s_w + j] = __ldg(&input[((bz * (size_t)in_h + ih) * in_w + iw)]);
            } else {
                shared_input[i * s_w + j] = -FLT_MAX;
            }
        }
    }
    __syncthreads();

    // Perform pooling
    #pragma unroll
    for (int my = 0; my < MULT_Y; ++my) {
        #pragma unroll
        for (int mx = 0; mx < MULT_X; ++mx) {
            int curr_out_x = out_x + mx * TILE_W;
            int curr_out_y = out_y + my * TILE_H;

            if (curr_out_y < out_h && curr_out_x < out_w) {
                int in_start_y = (curr_out_y * stride) - padding;
                int in_start_x = (curr_out_x * stride) - padding;
                
                float max_val = -FLT_MAX;
                #pragma unroll
                for (int i = 0; i < k_size; ++i) {
                    int row_idx = (in_start_y + i - start_y) * s_w;
                    #pragma unroll
                    for (int j = 0; j < k_size; ++j) {
                        max_val = fmaxf(max_val, shared_input[row_idx + (in_start_x + j - start_x)]);
                    }
                }
                output[((bz * (size_t)out_h + curr_out_y) * out_w + curr_out_x)] = max_val;
            }
        }
    }
}

void launch_max_pool2d(const torch::Tensor& input, torch::Tensor& output, 
                       int k_size, int stride, int padding) {
    const int batch = input.size(0);
    const int channels = input.size(1);
    const int in_h = input.size(2);
    const int in_w = input.size(3);
    const int out_h = output.size(2);
    const int out_w = output.size(3);

    dim3 block(TILE_W, TILE_H);
    dim3 grid((out_w + (TILE_W * MULT_X) - 1) / (TILE_W * MULT_X),
              (out_h + (TILE_H * MULT_Y) - 1) / (TILE_H * MULT_Y),
              batch * channels);

    int s_w = (TILE_W * MULT_X) * stride + k_size - 1;
    int s_h = (TILE_H * MULT_Y) * stride + k_size - 1;
    size_t shared_mem_size = s_w * s_h * sizeof(float);

    max_pool2d_kernel<<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        batch, channels, in_h, in_w, out_h, out_w,
        k_size, stride, padding
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void launch_max_pool2d(const torch::Tensor& input, torch::Tensor& output, int k_size, int stride, int padding);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &launch_max_pool2d, "Optimized MaxPool2D Forward");
}
"""

module = load_inline(
    name='max_pool2d_v2',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, maxpool_kernel_size, maxpool_stride, maxpool_padding, maxpool_dilation, maxpool_ceil_mode, maxpool_return_indices):
    if maxpool_dilation != 1 or maxpool_ceil_mode or maxpool_return_indices:
        raise NotImplementedError("Only basic MaxPool2D supported.")
    
    h_in, w_in = x.shape[2], x.shape[3]
    h_out = (h_in + 2 * maxpool_padding - maxpool_kernel_size) // maxpool_stride + 1
    w_out = (w_in + 2 * maxpool_padding - maxpool_kernel_size) // maxpool_stride + 1
    
    output = torch.empty((x.shape[0], x.shape[1], h_out, w_out), device=x.device, dtype=x.dtype)
    module.forward(x.contiguous(), output, maxpool_kernel_size, maxpool_stride, maxpool_padding)
    return output
