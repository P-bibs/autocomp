# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_063920/code_10.py
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
# CUDA kernel – optimized with hoisted index calculations and specialization
# -------------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16
#define MULT_FACTOR 3
#define OUTPUT_TILE (TILE_SIZE * MULT_FACTOR)

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

    const int shared_dim = OUTPUT_TILE * stride + k_size - 1;
    const int total_threads = TILE_SIZE * TILE_SIZE;
    const int thread_id = ty * TILE_SIZE + tx;
    const int tile_elements = shared_dim * shared_dim;

    const int ih_start = blockIdx.y * OUTPUT_TILE * stride - padding;
    const int iw_start = blockIdx.x * OUTPUT_TILE * stride - padding;

    // Coalesced load into shared memory
    for (int i = thread_id; i < tile_elements; i += total_threads) {
        const int row = i / shared_dim;
        const int col = i % shared_dim;
        const int ih = ih_start + row;
        const int iw = iw_start + col;

        if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
            shared_input[i] = input[((bz * (size_t)in_h + ih) * in_w + iw)];
        } else {
            shared_input[i] = -FLT_MAX;
        }
    }

    __syncthreads();

    // Precompute output base address for this thread
    const int base_output_offset = (bz * (size_t)out_h * out_w);
    
    // Hoist loop-invariant computations
    const int stride_times_tile = stride * TILE_SIZE;
    
    #pragma unroll
    for (int sub_y = 0; sub_y < MULT_FACTOR; ++sub_y) {
        // Cache sub_y computations
        const int sub_y_base_output = sub_y * TILE_SIZE;
        const int sub_y_base_shared = sub_y_base_output * stride;
        
        #pragma unroll
        for (int sub_x = 0; sub_x < MULT_FACTOR; ++sub_x) {
            const int ow = blockIdx.x * OUTPUT_TILE + sub_x * TILE_SIZE + tx;
            const int oh = blockIdx.y * OUTPUT_TILE + sub_y_base_output + ty;

            if (ow < out_w && oh < out_h) {
                // Hoist address calculations
                const int start_row = sub_y_base_shared + ty * stride;
                const int start_col = sub_x * stride_times_tile + tx * stride;
                const int base_shared_addr = start_row * shared_dim + start_col;
                
                float max_val = -FLT_MAX;

                // Specialize for common kernel sizes
                if (k_size == 2) {
                    const int row0_addr = start_row * shared_dim + start_col;
                    const int row1_addr = (start_row + 1) * shared_dim + start_col;
                    float v00 = shared_input[row0_addr];
                    float v01 = shared_input[row0_addr + 1];
                    float v10 = shared_input[row1_addr];
                    float v11 = shared_input[row1_addr + 1];
                    max_val = fmaxf(fmaxf(v00, v01), fmaxf(v10, v11));
                } else if (k_size == 3) {
                    const int row0_addr = start_row * shared_dim + start_col;
                    const int row1_addr = (start_row + 1) * shared_dim + start_col;
                    const int row2_addr = (start_row + 2) * shared_dim + start_col;
                    float v00 = shared_input[row0_addr];
                    float v01 = shared_input[row0_addr + 1];
                    float v02 = shared_input[row0_addr + 2];
                    float v10 = shared_input[row1_addr];
                    float v11 = shared_input[row1_addr + 1];
                    float v12 = shared_input[row1_addr + 2];
                    float v20 = shared_input[row2_addr];
                    float v21 = shared_input[row2_addr + 1];
                    float v22 = shared_input[row2_addr + 2];
                    max_val = fmaxf(fmaxf(fmaxf(v00, v01), fmaxf(v02, v10)), 
                                    fmaxf(fmaxf(v11, v12), fmaxf(v20, fmaxf(v21, v22))));
                } else if (k_size == 4) {
                    const int row0_addr = start_row * shared_dim + start_col;
                    const int row1_addr = (start_row + 1) * shared_dim + start_col;
                    const int row2_addr = (start_row + 2) * shared_dim + start_col;
                    const int row3_addr = (start_row + 3) * shared_dim + start_col;
                    
                    float vals[4][4];
                    vals[0][0] = shared_input[row0_addr];
                    vals[0][1] = shared_input[row0_addr + 1];
                    vals[0][2] = shared_input[row0_addr + 2];
                    vals[0][3] = shared_input[row0_addr + 3];
                    vals[1][0] = shared_input[row1_addr];
                    vals[1][1] = shared_input[row1_addr + 1];
                    vals[1][2] = shared_input[row1_addr + 2];
                    vals[1][3] = shared_input[row1_addr + 3];
                    vals[2][0] = shared_input[row2_addr];
                    vals[2][1] = shared_input[row2_addr + 1];
                    vals[2][2] = shared_input[row2_addr + 2];
                    vals[2][3] = shared_input[row2_addr + 3];
                    vals[3][0] = shared_input[row3_addr];
                    vals[3][1] = shared_input[row3_addr + 1];
                    vals[3][2] = shared_input[row3_addr + 2];
                    vals[3][3] = shared_input[row3_addr + 3];
                    
                    float max0 = fmaxf(fmaxf(vals[0][0], vals[0][1]), fmaxf(vals[0][2], vals[0][3]));
                    float max1 = fmaxf(fmaxf(vals[1][0], vals[1][1]), fmaxf(vals[1][2], vals[1][3]));
                    float max2 = fmaxf(fmaxf(vals[2][0], vals[2][1]), fmaxf(vals[2][2], vals[2][3]));
                    float max3 = fmaxf(fmaxf(vals[3][0], vals[3][1]), fmaxf(vals[3][2], vals[3][3]));
                    max_val = fmaxf(fmaxf(max0, max1), fmaxf(max2, max3));
                } else if (k_size == 5) {
                    const int row0_addr = start_row * shared_dim + start_col;
                    const int row1_addr = (start_row + 1) * shared_dim + start_col;
                    const int row2_addr = (start_row + 2) * shared_dim + start_col;
                    const int row3_addr = (start_row + 3) * shared_dim + start_col;
                    const int row4_addr = (start_row + 4) * shared_dim + start_col;
                    
                    float vals[5][5];
                    vals[0][0] = shared_input[row0_addr];
                    vals[0][1] = shared_input[row0_addr + 1];
                    vals[0][2] = shared_input[row0_addr + 2];
                    vals[0][3] = shared_input[row0_addr + 3];
                    vals[0][4] = shared_input[row0_addr + 4];
                    vals[1][0] = shared_input[row1_addr];
                    vals[1][1] = shared_input[row1_addr + 1];
                    vals[1][2] = shared_input[row1_addr + 2];
                    vals[1][3] = shared_input[row1_addr + 3];
                    vals[1][4] = shared_input[row1_addr + 4];
                    vals[2][0] = shared_input[row2_addr];
                    vals[2][1] = shared_input[row2_addr + 1];
                    vals[2][2] = shared_input[row2_addr + 2];
                    vals[2][3] = shared_input[row2_addr + 3];
                    vals[2][4] = shared_input[row2_addr + 4];
                    vals[3][0] = shared_input[row3_addr];
                    vals[3][1] = shared_input[row3_addr + 1];
                    vals[3][2] = shared_input[row3_addr + 2];
                    vals[3][3] = shared_input[row3_addr + 3];
                    vals[3][4] = shared_input[row3_addr + 4];
                    vals[4][0] = shared_input[row4_addr];
                    vals[4][1] = shared_input[row4_addr + 1];
                    vals[4][2] = shared_input[row4_addr + 2];
                    vals[4][3] = shared_input[row4_addr + 3];
                    vals[4][4] = shared_input[row4_addr + 4];
                    
                    float max0 = fmaxf(fmaxf(fmaxf(vals[0][0], vals[0][1]), fmaxf(vals[0][2], vals[0][3])), vals[0][4]);
                    float max1 = fmaxf(fmaxf(fmaxf(vals[1][0], vals[1][1]), fmaxf(vals[1][2], vals[1][3])), vals[1][4]);
                    float max2 = fmaxf(fmaxf(fmaxf(vals[2][0], vals[2][1]), fmaxf(vals[2][2], vals[2][3])), vals[2][4]);
                    float max3 = fmaxf(fmaxf(fmaxf(vals[3][0], vals[3][1]), fmaxf(vals[3][2], vals[3][3])), vals[3][4]);
                    float max4 = fmaxf(fmaxf(fmaxf(vals[4][0], vals[4][1]), fmaxf(vals[4][2], vals[4][3])), vals[4][4]);
                    max_val = fmaxf(fmaxf(fmaxf(max0, max1), fmaxf(max2, max3)), max4);
                } else {
                    // Generic path with hoisted row address computation
                    for (int i = 0; i < k_size; ++i) {
                        int row_idx = (start_row + i) * shared_dim;
                        for (int j = 0; j < k_size; ++j) {
                            max_val = fmaxf(max_val, shared_input[row_idx + start_col + j]);
                        }
                    }
                }
                
                // Cache output address computation
                const int output_idx = base_output_offset + oh * out_w + ow;
                output[output_idx] = max_val;
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

    const dim3 block(TILE_SIZE, TILE_SIZE);
    const dim3 grid((out_w + OUTPUT_TILE - 1) / OUTPUT_TILE,
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

cpp_source = r"""
#include <torch/extension.h>
void launch_max_pool2d(const torch::Tensor& input, torch::Tensor& output, int k_size, int stride, int padding);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &launch_max_pool2d, "Optimized MaxPool2D Forward");
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
        raise NotImplementedError("Only basic MaxPool2D supported.")
    
    h_in, w_in = x.shape[2], x.shape[3]
    h_out = (h_in + 2 * maxpool_padding - maxpool_kernel_size) // maxpool_stride + 1
    w_out = (w_in + 2 * maxpool_padding - maxpool_kernel_size) // maxpool_stride + 1
    
    output = torch.empty((x.shape[0], x.shape[1], h_out, w_out), device=x.device, dtype=x.dtype)
    module.forward(x.contiguous(), output, maxpool_kernel_size, maxpool_stride, maxpool_padding)
    return output
