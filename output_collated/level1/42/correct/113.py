# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_063920/code_8.py
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
#  CUDA kernel – Optimized Max-pool 2D with register caching
#  Key Optimizations:
#  1. Pre-compute base pointer to eliminate repeated 2D indexing
#  2. Register caching of kernel window to reduce shared memory accesses
#  3. Strided access pattern for better memory coalescing
#  4. Efficient reduction using fmaxf with register-local data
# ----------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <algorithm>

#define TILE_SIZE 16
#define MULT_FACTOR 2
#define OUTPUT_TILE (TILE_SIZE * MULT_FACTOR)
#define MIN_VAL -1e38f
#define MAX_KERNEL_SIZE 16

__global__ void max_pool2d_kernel_optimized(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size, int channels,
    int in_h, int in_w,
    int out_h, int out_w,
    int k_size, int stride, int padding) {

    extern __shared__ float shared_mem[];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bz = blockIdx.z;

    const int shared_dim = OUTPUT_TILE * stride + k_size - 1;
    const int thread_id = ty * TILE_SIZE + tx;
    const int total_threads = TILE_SIZE * TILE_SIZE;
    const int region_size = shared_dim * shared_dim;

    int ih_start = blockIdx.y * OUTPUT_TILE * stride - padding;
    int iw_start = blockIdx.x * OUTPUT_TILE * stride - padding;

    // Load input tiles into shared memory without divergent branches
    for (int i = thread_id; i < region_size; i += total_threads) {
        int row = i / shared_dim;
        int col = i % shared_dim;
        int ih = ih_start + row;
        int iw = iw_start + col;

        bool valid = (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w);
        shared_mem[i] = valid ? input[((bz * in_h + ih) * in_w + iw)] : MIN_VAL;
    }

    __syncthreads();

    // Compute max pooling with optimized pointer arithmetic
    #pragma unroll
    for (int sub_y = 0; sub_y < MULT_FACTOR; ++sub_y) {
        #pragma unroll
        for (int sub_x = 0; sub_x < MULT_FACTOR; ++sub_x) {
            int ow = blockIdx.x * OUTPUT_TILE + sub_x * TILE_SIZE + tx;
            int oh = blockIdx.y * OUTPUT_TILE + sub_y * TILE_SIZE + ty;

            if (oh < out_h && ow < out_w) {
                float max_val = MIN_VAL;
                
                // Pre-compute base pointer to kernel window in shared memory
                int start_row = (sub_y * TILE_SIZE + ty) * stride;
                int start_col = (sub_x * TILE_SIZE + tx) * stride;
                int base_idx = start_row * shared_dim + start_col;

                // Load kernel window into register cache
                float kernel_vals[MAX_KERNEL_SIZE];
                int val_count = 0;

                #pragma unroll
                for (int ki = 0; ki < k_size; ++ki) {
                    int row_idx = base_idx + ki * shared_dim;
                    #pragma unroll
                    for (int kj = 0; kj < k_size; ++kj) {
                        kernel_vals[val_count++] = shared_mem[row_idx + kj];
                    }
                }

                // Perform reduction on register-cached values
                #pragma unroll
                for (int i = 0; i < k_size * k_size; ++i) {
                    max_val = fmaxf(max_val, kernel_vals[i]);
                }

                output[((bz * out_h + oh) * out_w + ow)] = max_val;
            }
        }
    }
}

void launch_max_pool2d_optimized(
    const torch::Tensor& input, 
    torch::Tensor& output, 
    int k_size, int stride, int padding) {
    
    int batch = input.size(0);
    int channels = input.size(1);
    int in_h = input.size(2);
    int in_w = input.size(3);
    int out_h = output.size(2);
    int out_w = output.size(3);

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((out_w + OUTPUT_TILE - 1) / OUTPUT_TILE,
              (out_h + OUTPUT_TILE - 1) / OUTPUT_TILE,
              batch * channels);

    const int shared_dim = OUTPUT_TILE * stride + k_size - 1;
    size_t shared_mem_size = shared_dim * shared_dim * sizeof(float);

    max_pool2d_kernel_optimized<<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        batch, channels, in_h, in_w, out_h, out_w,
        k_size, stride, padding
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void launch_max_pool2d_optimized(
    const torch::Tensor& input, 
    torch::Tensor& output, 
    int k_size, int stride, int padding);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &launch_max_pool2d_optimized, "Optimized Max Pool 2D with Register Caching");
}
"""

module = load_inline(
    name='max_pool2d_opt', 
    cpp_sources=cpp_source, 
    cuda_sources=cuda_kernel, 
    extra_cuda_cflags=['-O3', '--use_fast_math', '-Xptxas', '-O3'], 
    with_cuda=True
)

def functional_model(x, *, maxpool_kernel_size, maxpool_stride, maxpool_padding, 
                    maxpool_dilation, maxpool_ceil_mode, maxpool_return_indices):
    if maxpool_dilation != 1 or maxpool_ceil_mode or maxpool_return_indices:
        raise NotImplementedError("Only standard max pooling is supported.")
        
    h_in, w_in = x.shape[2], x.shape[3]
    h_out = (h_in + 2 * maxpool_padding - maxpool_kernel_size) // maxpool_stride + 1
    w_out = (w_in + 2 * maxpool_padding - maxpool_kernel_size) // maxpool_stride + 1
    
    output = torch.empty(
        (x.shape[0], x.shape[1], h_out, w_out), 
        device=x.device, 
        dtype=x.dtype
    )
    module.forward(
        x.contiguous(), output, 
        maxpool_kernel_size, maxpool_stride, maxpool_padding
    )
    return output
