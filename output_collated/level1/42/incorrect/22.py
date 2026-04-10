# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_063920/code_9.py
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

# Optimization rationale:
# 1. float4 vectorization: Loads 4 floats per memory transaction, maximizing bandwidth
#    utilization on RTX 2080Ti (616 GB/s peak requires wide operations)
# 2. __ldg intrinsics: Combined with vectorization for L1 read-only cache efficiency
# 3. Tiling strategy: 3x3 tiles per thread block to maximize shared memory
# 4. Memory Coalescing: Vectorized cooperative loads ensure optimal coalescing

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>

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
    
    // Vectorized coalesced load into shared memory using float4
    const int vec_tile_elements = (tile_elements + 3) / 4;
    for (int i = thread_id; i < vec_tile_elements; i += total_threads) {
        const int base_idx = i * 4;
        const float4* input_vec = reinterpret_cast<const float4*>(input);
        
        // Calculate base coordinates for all 4 elements
        int rows[4], cols[4];
        #pragma unroll
        for (int v = 0; v < 4; ++v) {
            const int idx = base_idx + v;
            if (idx < tile_elements) {
                rows[v] = idx / shared_dim;
                cols[v] = idx % shared_dim;
            }
        }
        
        // Vectorized load if aligned and within bounds
        float4 data = make_float4(-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX);
        bool can_vectorize = true;
        
        // Check alignment and contiguity
        for (int v = 0; v < 4; ++v) {
            const int idx = base_idx + v;
            if (idx >= tile_elements) break;
            
            const int ih = ih_start + rows[v];
            const int iw = iw_start + cols[v];
            
            if (!(ih >= 0 && ih < in_h && iw >= 0 && iw < in_w)) {
                can_vectorize = false;
                break;
            }
        }
        
        if (can_vectorize && base_idx + 3 < tile_elements) {
            // All 4 elements are valid and contiguous - try vectorized load
            const int ih0 = ih_start + rows[0];
            const int iw0 = iw_start + cols[0];
            const size_t base_input_idx = ((bz * (size_t)in_h + ih0) * in_w + iw0);
            
            // Check if we can do a 4-element vectorized read
            if (base_input_idx % 4 == 0 && iw0 + 3 < in_w) {
                data = __ldg(&input_vec[base_input_idx / 4]);
            } else {
                can_vectorize = false;
            }
        } else {
            can_vectorize = false;
        }
        
        // Store the data (vectorized or scalar)
        if (can_vectorize) {
            shared_input[base_idx]     = data.x;
            shared_input[base_idx + 1] = data.y;
            shared_input[base_idx + 2] = data.z;
            shared_input[base_idx + 3] = data.w;
        } else {
            #pragma unroll 4
            for (int v = 0; v < 4; ++v) {
                const int idx = base_idx + v;
                if (idx < tile_elements) {
                    const int row = rows[v];
                    const int col = cols[v];
                    const int ih = ih_start + row;
                    const int iw = iw_start + col;
                    
                    if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                        shared_input[idx] = __ldg(&input[((bz * (size_t)in_h + ih) * in_w + iw)]);
                    } else {
                        shared_input[idx] = -FLT_MAX;
                    }
                }
            }
        }
    }

    __syncthreads();

    #pragma unroll
    for (int sub_y = 0; sub_y < MULT_FACTOR; ++sub_y) {
        #pragma unroll
        for (int sub_x = 0; sub_x < MULT_FACTOR; ++sub_x) {
            const int ow = blockIdx.x * OUTPUT_TILE + sub_x * TILE_SIZE + tx;
            const int oh = blockIdx.y * OUTPUT_TILE + sub_y * TILE_SIZE + ty;

            if (ow < out_w && oh < out_h) {
                float max_val = -FLT_MAX;
                const int my_start_row = (sub_y * TILE_SIZE + ty) * stride;
                const int my_start_col = (sub_x * TILE_SIZE + tx) * stride;

                if (k_size == 2) {
                    float v1 = shared_input[(my_start_row + 0) * shared_dim + my_start_col + 0];
                    float v2 = shared_input[(my_start_row + 0) * shared_dim + my_start_col + 1];
                    float v3 = shared_input[(my_start_row + 1) * shared_dim + my_start_col + 0];
                    float v4 = shared_input[(my_start_row + 1) * shared_dim + my_start_col + 1];
                    max_val = fmaxf(v1, fmaxf(v2, fmaxf(v3, v4)));
                } else {
                    #pragma unroll
                    for (int i = 0; i < k_size; ++i) {
                        int row_idx = (my_start_row + i) * shared_dim;
                        #pragma unroll
                        for (int j = 0; j < k_size; ++j) {
                            max_val = fmaxf(max_val, shared_input[row_idx + my_start_col + j]);
                        }
                    }
                }
                output[((bz * (size_t)out_h + oh) * out_w + ow)] = max_val;
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
    m.def("forward", &launch_max_pool2d, "Vectorized MaxPool2D Forward");
}
"""

# Compilation with aggressive optimizations
module = load_inline(
    name='max_pool2d_vectorized',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math', '-Xptxas=-v'],
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
