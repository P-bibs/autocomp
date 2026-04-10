# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_054839/code_30.py
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
# CUDA source – max-pool 2-D with optimized coalesced tile loading
# -------------------------------------------------------------------------
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

    // Dynamic shared memory – holds one tile of the input
    extern __shared__ float shared_mem[];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bz = blockIdx.z;

    int tile_dim = TILE_SIZE * stride + k_size - 1;

    // Top-left corner of the input tile corresponding to the output block
    int ih_start = blockIdx.y * TILE_SIZE * stride - padding;
    int iw_start = blockIdx.x * TILE_SIZE * stride - padding;

    // ---------------------------------------------------------
    // Optimized loading pattern: nested loop for global coalescing
    // ---------------------------------------------------------
    // Threads collaborate to load the tile into shared memory.
    // By mapping tx to the column index (inner loop), we ensure 
    // contiguous memory access for coalescing.
    for (int r = ty; r < tile_dim; r += blockDim.y) {
        int global_row = ih_start + r;
        for (int c = tx; c < tile_dim; c += blockDim.x) {
            int global_col = iw_start + c;
            int sIdx = r * tile_dim + c;
            if (global_row >= 0 && global_row < in_h &&
                global_col >= 0 && global_col < in_w) {
                shared_mem[sIdx] = input[((bz * in_h + global_row) * in_w + global_col)];
            } else {
                shared_mem[sIdx] = -1e38f;
            }
        }
    }

    __syncthreads();

    // ---------------------------------------------------------
    // Compute max-pool
    // ---------------------------------------------------------
    int ow = blockIdx.x * TILE_SIZE + tx;
    int oh = blockIdx.y * TILE_SIZE + ty;

    if (ow < out_w && oh < out_h) {
        float max_val = -1e38f;
        int start_r = ty * stride;
        int start_c = tx * stride;

        for (int i = 0; i < k_size; ++i) {
            int row_offset = (start_r + i) * tile_dim;
            for (int j = 0; j < k_size; ++j) {
                float val = shared_mem[row_offset + start_c + j];
                if (val > max_val) max_val = val;
            }
        }
        output[((bz * out_h + oh) * out_w + ow)] = max_val;
    }
}

void max_pool2d_forward_impl(const torch::Tensor& input, torch::Tensor& output,
                             int k_size, int stride, int padding) {
    int batch = input.size(0);
    int channels = input.size(1);
    int in_h = input.size(2);
    int in_w = input.size(3);
    int out_h = output.size(2);
    int out_w = output.size(3);

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((out_w + TILE_SIZE - 1) / TILE_SIZE,
              (out_h + TILE_SIZE - 1) / TILE_SIZE,
              batch * channels);

    int tile_dim = TILE_SIZE * stride + k_size - 1;
    size_t shared_mem_sz = tile_dim * tile_dim * sizeof(float);

    max_pool2d_kernel<<<grid, block, shared_mem_sz>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        batch, channels, in_h, in_w, out_h, out_w,
        k_size, stride, padding
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void max_pool2d_forward_impl(const torch::Tensor& input, torch::Tensor& output, int k_size, int stride, int padding);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_forward_impl, "Max Pool 2D Forward Optimized");
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
        raise NotImplementedError("Only standard MaxPool2D (dilation=1, ceil_mode=False, indices=False) supported.")
    
    h_in, w_in = x.shape[2], x.shape[3]
    h_out = (h_in + 2 * maxpool_padding - maxpool_kernel_size) // maxpool_stride + 1
    w_out = (w_in + 2 * maxpool_padding - maxpool_kernel_size) // maxpool_stride + 1
    
    output = torch.empty((x.shape[0], x.shape[1], h_out, w_out), device=x.device, dtype=x.dtype)
    module.forward(x.contiguous(), output, maxpool_kernel_size, maxpool_stride, maxpool_padding)
    return output
