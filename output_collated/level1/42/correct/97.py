# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_062618/code_14.py
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
#  CUDA kernel – max‑pool 2D with branch‑less boundary handling
# ----------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16
#define MULT_FACTOR 2
#define OUTPUT_TILE (TILE_SIZE * MULT_FACTOR)
#define VEC_SIZE 4

// Kernel: vectorised load, branch‑less boundary checks, shared‑memory tiled pooling
__global__ void max_pool2d_kernel_vectorized(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size, int channels,
    int in_h, int in_w,
    int out_h, int out_w,
    int k_size, int stride, int padding) {

    // Shared memory – we will reinterpret it as float4 for vectorised stores
    extern __shared__ float shared_mem[];
    float4* shared_input = reinterpret_cast<float4*>(shared_mem);

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bz = blockIdx.z;                     // (batch * channels + channel)

    const int shared_dim = OUTPUT_TILE * stride + k_size - 1;
    const int thread_id   = ty * blockDim.x + tx;
    const int total_threads = blockDim.x * blockDim.y;

    // Upper‑left corner of the region this block reads from the input
    int ih_start = blockIdx.y * OUTPUT_TILE * stride - padding;
    int iw_start = blockIdx.x * OUTPUT_TILE * stride - padding;

    // Total number of elements that must be resident in shared memory
    const int region_size = shared_dim * shared_dim;
    // Number of float4 vectors that cover the region (rounded up)
    const int vec_count = (region_size + VEC_SIZE - 1) / VEC_SIZE;

    // --------------------------------------------------------------
    // 1) Load the required input region into shared memory
    //    – one float4 per thread, vectorised, branch‑less mask
    // --------------------------------------------------------------
    for (int i = thread_id; i < vec_count; i += total_threads) {
        int base_idx = i * VEC_SIZE;
        // Initialise all components with -infinity
        float4 val = make_float4(-1e38f, -1e38f, -1e38f, -1e38f);

        #pragma unroll
        for (int v = 0; v < VEC_SIZE; ++v) {
            int idx = base_idx + v;
            if (idx < region_size) {
                int row = idx / shared_dim;
                int col = idx % shared_dim;
                int ih  = ih_start + row;
                int iw  = iw_start + col;

                // Boundary test as a predicate (no divergent branch)
                bool valid = (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w);
                float load_val = valid ? input[((bz * in_h + ih) * in_w + iw)] : -1e38f;

                // Write into the appropriate component of the vector
                switch (v) {
                    case 0: val.x = load_val; break;
                    case 1: val.y = load_val; break;
                    case 2: val.z = load_val; break;
                    case 3: val.w = load_val; break;
                }
            }
        }
        shared_input[i] = val;
    }

    __syncthreads();

    // --------------------------------------------------------------
    // 2) Compute max‑pooling for the output points owned by this thread
    // --------------------------------------------------------------
    #pragma unroll
    for (int sub_y = 0; sub_y < MULT_FACTOR; ++sub_y) {
        #pragma unroll
        for (int sub_x = 0; sub_x < MULT_FACTOR; ++sub_x) {
            int ow = blockIdx.x * OUTPUT_TILE + sub_x * TILE_SIZE + tx;
            int oh = blockIdx.y * OUTPUT_TILE + sub_y * TILE_SIZE + ty;

            if (ow < out_w && oh < out_h) {
                int start_row = (sub_y * TILE_SIZE + ty) * stride;
                int start_col = (sub_x * TILE_SIZE + tx) * stride;

                float max_val = -1e38f;

                // Special‑case k_size == 2 (fully unrolled)
                if (k_size == 2) {
                    float v00 = shared_mem[(start_row + 0) * shared_dim + start_col + 0];
                    float v01 = shared_mem[(start_row + 0) * shared_dim + start_col + 1];
                    float v10 = shared_mem[(start_row + 1) * shared_dim + start_col + 0];
                    float v11 = shared_mem[(start_row + 1) * shared_dim + start_col + 1];
                    max_val = fmaxf(fmaxf(v00, v01), fmaxf(v10, v11));
                } else {
                    // Generic kernel – unroll the innermost loops when possible
                    #pragma unroll
                    for (int ki = 0; ki < k_size; ++ki) {
                        #pragma unroll
                        for (int kj = 0; kj < k_size; ++kj) {
                            float v = shared_mem[(start_row + ki) * shared_dim + start_col + kj];
                            max_val = fmaxf(max_val, v);
                        }
                    }
                }

                output[((bz * out_h + oh) * out_w + ow)] = max_val;
            }
        }
    }
}

// Host function that launches the kernel
void max_pool2d_forward(const torch::Tensor& input,
                        torch::Tensor& output,
                        int k_size, int stride, int padding) {
    int batch    = input.size(0);
    int channels = input.size(1);
    int in_h     = input.size(2);
    int in_w     = input.size(3);
    int out_h    = output.size(2);
    int out_w    = output.size(3);

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((out_w + OUTPUT_TILE - 1) / OUTPUT_TILE,
              (out_h + OUTPUT_TILE - 1) / OUTPUT_TILE,
              batch * channels);

    const int shared_dim = OUTPUT_TILE * stride + k_size - 1;
    size_t shared_mem_size = static_cast<size_t>(shared_dim) * shared_dim * sizeof(float);

    max_pool2d_kernel_vectorized<<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch, channels, in_h, in_w, out_h, out_w,
        k_size, stride, padding);
}
"""

# ----------------------------------------------------------------------
#  C++ binding (pybind11)
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void max_pool2d_forward(const torch::Tensor& input,
                        torch::Tensor& output,
                        int k_size, int stride, int padding);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_forward,
          "Vectorized Max Pool 2D forward (CUDA)");
}
"""

# ----------------------------------------------------------------------
#  Compile the inline CUDA extension
# ----------------------------------------------------------------------
module = load_inline(
    name='max_pool2d_vec',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# ----------------------------------------------------------------------
#  Functional wrapper expected by the evaluation harness
# ----------------------------------------------------------------------
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
    """
    Performs 2D max‑pooling using a custom CUDA kernel.
    Only the standard case (dilation=1, ceil_mode=False, return_indices=False)
    is supported; other settings raise an error.
    """
    if maxpool_dilation != 1:
        raise NotImplementedError("Dilation != 1 is not supported in this kernel.")
    if maxpool_ceil_mode:
        raise NotImplementedError("Ceil mode is not supported in this kernel.")
    if maxpool_return_indices:
        raise NotImplementedError("Returning indices is not supported in this kernel.")

    # Compute output spatial size (same formula as torch.nn.functional.max_pool2d)
    h_in, w_in = x.shape[2], x.shape[3]
    h_out = (h_in + 2 * maxpool_padding - maxpool_kernel_size) // maxpool_stride + 1
    w_out = (w_in + 2 * maxpool_padding - maxpool_kernel_size) // maxpool_stride + 1

    # Allocate output tensor
    output = torch.empty(
        (x.shape[0], x.shape[1], h_out, w_out),
        device=x.device,
        dtype=x.dtype
    )

    # Invoke the compiled CUDA kernel
    module.forward(
        x.contiguous(),
        output,
        maxpool_kernel_size,
        maxpool_stride,
        maxpool_padding
    )
    return output
