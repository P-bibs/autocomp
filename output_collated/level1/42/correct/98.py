# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_062618/code_15.py
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

# ----------------------------------------------------------------------
# Optimised CUDA kernel – uses __ldg, -INFINITY and #pragma unroll
# ----------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>

#define TILE_SIZE 16

// Negative infinity sentinel – the hardware can treat this as a true -inf
#define NEG_INF (-INFINITY)

__global__ void max_pool2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size, int channels,
    int in_h, int in_w,
    int out_h, int out_w,
    int k_size, int stride, int padding) {

    // Dynamically sized shared memory (same size formula as before)
    extern __shared__ float shared_input[];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * TILE_SIZE + tx;
    const int bz = blockIdx.z;

    // Tile configuration – MULT_FACTOR = 2 yields a 32×32 output tile per block
    const int MULT_FACTOR = 2;
    const int OUTPUT_TILE = TILE_SIZE * MULT_FACTOR;
    const int shared_dim = OUTPUT_TILE * stride + k_size - 1;
    const int total_threads = TILE_SIZE * TILE_SIZE;
    const int tile_elements = shared_dim * shared_dim;

    // Starting coordinates of the current tile in the input (including padding)
    const int ih_start = blockIdx.y * OUTPUT_TILE * stride - padding;
    const int iw_start = blockIdx.x * OUTPUT_TILE * stride - padding;

    // ------------------------------------------------------------------
    // 1) Load the input tile into shared memory
    //    * Use __ldg() for read‑only cache traffic
    //    * Coalesced access pattern (same as original)
    // ------------------------------------------------------------------
    for (int i = tid; i < tile_elements; i += total_threads) {
        const int row = i / shared_dim;
        const int col = i % shared_dim;
        const int ih = ih_start + row;
        const int iw = iw_start + col;

        // Boundary check – all conditions are cheap
        const bool valid = (ih >= 0) & (ih < in_h) & (iw >= 0) & (iw < in_w);
        // Use __ldg for the global load and a true -INFINITY for invalid entries
        const int input_idx = (bz * in_h + ih) * in_w + iw;
        shared_input[i] = valid ? __ldg(&input[input_idx]) : NEG_INF;
    }

    __syncthreads();

    // ------------------------------------------------------------------
    // 2) Compute the pooling results
    //    * Unroll the two sub‑tile loops (MULT_FACTOR = 2)
    //    * Use a fast unrolled path for k_size == 2
    //    * Use -INFINITY sentinel in the generic path as well
    // ------------------------------------------------------------------
    #pragma unroll
    for (int sub_y = 0; sub_y < MULT_FACTOR; ++sub_y) {
        #pragma unroll
        for (int sub_x = 0; sub_x < MULT_FACTOR; ++sub_x) {
            const int ow = blockIdx.x * OUTPUT_TILE + sub_x * TILE_SIZE + tx;
            const int oh = blockIdx.y * OUTPUT_TILE + sub_y * TILE_SIZE + ty;

            if (ow < out_w && oh < out_h) {
                const int start_row = (sub_y * TILE_SIZE + ty) * stride;
                const int start_col = (sub_x * TILE_SIZE + tx) * stride;

                float max_val = NEG_INF;

                if (k_size == 2) {
                    // Fully unrolled 2×2 pooling – minimal instructions
                    const int base_idx1 = (start_row + 0) * shared_dim + start_col;
                    const int base_idx2 = (start_row + 1) * shared_dim + start_col;

                    const float v1 = shared_input[base_idx1 + 0];
                    const float v2 = shared_input[base_idx1 + 1];
                    const float v3 = shared_input[base_idx2 + 0];
                    const float v4 = shared_input[base_idx2 + 1];

                    // fmaxf is fast; the chain of three calls is fully unrolled
                    max_val = fmaxf(fmaxf(fmaxf(v1, v2), v3), v4);
                } else {
                    // Generic pooling loop – also unroll the inner loop when possible
                    #pragma unroll 7
                    for (int i = 0; i < 7; ++i) {          // max kernel size in typical models
                        if (i >= k_size) break;
                        const int row_offset = (start_row + i) * shared_dim + start_col;
                        #pragma unroll 7
                        for (int j = 0; j < 7; ++j) {
                            if (j >= k_size) break;
                            const float v = shared_input[row_offset + j];
                            max_val = fmaxf(max_val, v);
                        }
                    }
                }

                output[((bz * out_h + oh) * out_w + ow)] = max_val;
            }
        }
    }
}

void max_pool2d_forward(const torch::Tensor& input, torch::Tensor& output,
                        int k_size, int stride, int padding) {
    const int batch   = input.size(0);
    const int channels = input.size(1);
    const int in_h    = input.size(2);
    const int in_w    = input.size(3);
    const int out_h   = output.size(2);
    const int out_w   = output.size(3);

    const int MULT_FACTOR = 2;
    const int OUTPUT_TILE = TILE_SIZE * MULT_FACTOR;

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((out_w + OUTPUT_TILE - 1) / OUTPUT_TILE,
              (out_h + OUTPUT_TILE - 1) / OUTPUT_TILE,
              batch * channels);

    const int shared_dim = OUTPUT_TILE * stride + k_size - 1;
    const size_t shared_mem_size = shared_dim * shared_dim * sizeof(float);

    max_pool2d_kernel<<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        batch, channels, in_h, in_w, out_h, out_w,
        k_size, stride, padding
    );
}
"""

# --------------------------------------------------------------------------
# C++ binding (PYBIND11) – same signature as before
# --------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void max_pool2d_forward(const torch::Tensor& input, torch::Tensor& output,
                        int k_size, int stride, int padding);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_forward,
          "Max‑pool 2‑D forward (optimised with __ldg, -INFINITY and loop unrolling)");
}
"""

# --------------------------------------------------------------------------
# Build the inline extension
# --------------------------------------------------------------------------
module = load_inline(
    name='max_pool2d_optimized',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# --------------------------------------------------------------------------
# Public interface – same semantics as the original functional_model
# --------------------------------------------------------------------------
def functional_model(
    x: torch.Tensor,
    *,
    maxpool_kernel_size: int,
    maxpool_stride: int,
    maxpool_padding: int,
    maxpool_dilation: int,
    maxpool_ceil_mode: bool,
    maxpool_return_indices: bool
) -> torch.Tensor:
    """Performs a 2‑D max‑pooling using a custom CUDA kernel."""
    if maxpool_dilation != 1:
        raise NotImplementedError("Dilation is not supported in this kernel.")
    if maxpool_ceil_mode:
        raise NotImplementedError("Ceil mode is not supported.")
    if maxpool_return_indices:
        raise NotImplementedError("Returning indices is not supported.")

    # Compute output spatial size (same formula as PyTorch's nn.MaxPool2d)
    h_in, w_in = x.shape[2], x.shape[3]
    h_out = (h_in + 2 * maxpool_padding - maxpool_kernel_size) // maxpool_stride + 1
    w_out = (w_in + 2 * maxpool_padding - maxpool_kernel_size) // maxpool_stride + 1

    # Allocate output tensor on the same device/dtype as input
    output = torch.empty(
        (x.shape[0], x.shape[1], h_out, w_out),
        device=x.device,
        dtype=x.dtype
    )

    # Call the optimised CUDA implementation
    module.forward(
        x.contiguous(),
        output,
        maxpool_kernel_size,
        maxpool_stride,
        maxpool_padding
    )
    return output
