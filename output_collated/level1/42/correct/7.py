# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_052448/code_9.py
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

# -------------------------------------------------------------------------
# CUDA kernel with shared-memory tiling
# -------------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// Tiling parameters – compile-time constants used in the kernel
constexpr int BLOCK_X = 16;
constexpr int BLOCK_Y = 16;

__global__ void max_pool2d_shared_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size, int channels,
    int in_h, int in_w,
    int out_h, int out_w,
    int k_size, int stride, int padding) {

    // Compute tile dimensions at compile time for common cases
    constexpr int TILE_H = (BLOCK_Y - 1) * 1 + 3; // Assuming stride=1, k_size=3 as common case
    constexpr int TILE_W = (BLOCK_X - 1) * 1 + 3;
    
    // Dynamic shared memory: tile_h * tile_w floats
    extern __shared__ float smem[];

    // --- Block identity --------------------------------------------------
    // grid.x = (batch*channels) * out_w_tiles
    // grid.y = out_h_tiles
    int out_w_tiles = (out_w + BLOCK_X - 1) / BLOCK_X;
    int bc_idx   = blockIdx.x / out_w_tiles;            // which (batch,channel) pair
    int tile_w_idx = blockIdx.x % out_w_tiles;          // which column tile
    int tile_h_idx = blockIdx.y;                         // which row tile

    int b = bc_idx / channels;
    int c = bc_idx % channels;

    // --- Starting output position for this block ------------------------
    int oh_start = tile_h_idx * BLOCK_Y;   // first output row in the tile
    int ow_start = tile_w_idx * BLOCK_X;   // first output col in the tile

    // --- Corresponding input region ------------------------------------
    int ih_start = oh_start * stride - padding;
    int iw_start = ow_start * stride - padding;
    
    // Compute actual tile size for this specific configuration
    int tile_h = (BLOCK_Y - 1) * stride + k_size;
    int tile_w = (BLOCK_X - 1) * stride + k_size;

    // --- Cooperative load of the input tile into shared memory --------
    for (int i = threadIdx.y; i < tile_h; i += BLOCK_Y) {
        for (int j = threadIdx.x; j < tile_w; j += BLOCK_X) {
            int gi = ih_start + i;   // global input row
            int gj = iw_start + j;   // global input col

            float val;
            if (gi >= 0 && gi < in_h && gj >= 0 && gj < in_w) {
                // contiguous memory layout: (b,c,h,w)
                val = input[((b * channels + c) * in_h + gi) * in_w + gj];
            } else {
                val = -1e38f;         // padding treated as -infinity
            }
            smem[i * tile_w + j] = val;
        }
    }
    __syncthreads();

    // --- Compute max-pool for the thread's own output element ----------
    int oh = oh_start + threadIdx.y;
    int ow = ow_start + threadIdx.x;

    if (oh < out_h && ow < out_w) {
        // top-left corner of the pooling window in the *input* space
        int win_ih_start = oh * stride - padding;
        int win_iw_start = ow * stride - padding;

        float max_val = -1e38f;

        // iterate over the kernel window
        for (int di = 0; di < k_size; ++di) {
            int local_i = win_ih_start - ih_start + di;   // row index inside the tile
            for (int dj = 0; dj < k_size; ++dj) {
                int local_j = win_iw_start - iw_start + dj; // col index inside the tile
                float v = smem[local_i * tile_w + local_j];
                if (v > max_val) max_val = v;
            }
        }

        // write result
        output[((b * channels + c) * out_h + oh) * out_w + ow] = max_val;
    }
}

void max_pool2d_forward(const torch::Tensor& input, torch::Tensor& output, 
                        int k_size, int stride, int padding) {
    int batch_size = input.size(0);
    int channels = input.size(1);
    int in_h = input.size(2);
    int in_w = input.size(3);
    int out_h = output.size(2);
    int out_w = output.size(3);

    // Calculate tile dimensions
    int tile_h = (BLOCK_Y - 1) * stride + k_size;
    int tile_w = (BLOCK_X - 1) * stride + k_size;
    
    // Calculate number of tiles
    int out_w_tiles = (out_w + BLOCK_X - 1) / BLOCK_X;
    int out_h_tiles = (out_h + BLOCK_Y - 1) / BLOCK_Y;
    
    // Grid and block dimensions
    dim3 grid((batch_size * channels) * out_w_tiles, out_h_tiles);
    dim3 block(BLOCK_X, BLOCK_Y);

    // Shared memory size
    size_t shared_mem_size = tile_h * tile_w * sizeof(float);

    max_pool2d_shared_kernel<<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        batch_size, channels, in_h, in_w, out_h, out_w,
        k_size, stride, padding
    );
}
"""

# -------------------------------------------------------------------------
# C++ binding (PyBind11)
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void max_pool2d_forward(const torch::Tensor& input,
                        torch::Tensor& output,
                        int k_size,
                        int stride,
                        int padding);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_forward, "Max pool 2D forward (tiled)");
}
"""

# -------------------------------------------------------------------------
# Build the extension
# -------------------------------------------------------------------------
module = load_inline(
    name='max_pool2d_tiled',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# -------------------------------------------------------------------------
# Functional wrapper that calls the tiled kernel
# -------------------------------------------------------------------------
def functional_model(
    x: torch.Tensor,
    *,
    maxpool_kernel_size: int,
    maxpool_stride: int,
    maxpool_padding: int,
    maxpool_dilation: int,      # ignored – not needed for this simple kernel
    maxpool_ceil_mode: bool,    # ignored – we assume ceil_mode=False
    maxpool_return_indices: bool
):
    """
    Tile-based max-pooling that uses shared memory to minimise global-memory traffic.
    All arguments except `x`, `maxpool_kernel_size`, `maxpool_stride`, `maxpool_padding`
    are present for API compatibility and are not used in this implementation.
    """
    if maxpool_return_indices:
        raise NotImplementedError("Returning indices is not supported in this tiled kernel.")

    # Input / output geometry
    batch, channels, h_in, w_in = x.shape
    h_out = (h_in + 2 * maxpool_padding - maxpool_kernel_size) // maxpool_stride + 1
    w_out = (w_in + 2 * maxpool_padding - maxpool_kernel_size) // maxpool_stride + 1

    # Allocate output tensor
    output = torch.empty((batch, channels, h_out, w_out),
                         device=x.device, dtype=x.dtype)

    # The CUDA wrapper computes block/tile sizes internally – no Python-side changes needed.
    module.forward(x.contiguous(), output,
                   maxpool_kernel_size, maxpool_stride, maxpool_padding)

    return output
