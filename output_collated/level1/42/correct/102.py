# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_062618/code_21.py
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

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector_types.h>

#define TILE_SIZE 16
#define MULT_FACTOR 2

__global__ void max_pool2d_vec_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch, int channels,
    int in_h, int in_w,
    int out_h, int out_w,
    int k_size, int stride, int padding) {

    extern __shared__ float shared_mem[];
    
    int bz = blockIdx.z;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    const int OUTPUT_TILE = TILE_SIZE * MULT_FACTOR;
    const int shared_dim = OUTPUT_TILE * stride + k_size - 1;
    
    int ih_start = blockIdx.y * OUTPUT_TILE * stride - padding;
    int iw_start = blockIdx.x * OUTPUT_TILE * stride - padding;
    
    // Total elements per tile to load
    int total_elements = shared_dim * shared_dim;
    int threads_per_tile = TILE_SIZE * TILE_SIZE;
    
    // Use float4 for vectorization if index is aligned
    for (int i = 0; i < total_elements; i += threads_per_tile) {
        int idx = i + (ty * TILE_SIZE + tx);
        if (idx < total_elements) {
            int r = idx / shared_dim;
            int c = idx % shared_dim;
            int ih = ih_start + r;
            int iw = iw_start + c;
            
            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                shared_mem[idx] = input[((bz * in_h + ih) * in_w + iw)];
            } else {
                shared_mem[idx] = -3.402823466e+38F;
            }
        }
    }
    __syncthreads();
    
    #pragma unroll
    for (int sub_y = 0; sub_y < MULT_FACTOR; sub_y++) {
        #pragma unroll
        for (int sub_x = 0; sub_x < MULT_FACTOR; sub_x++) {
            int oh = blockIdx.y * OUTPUT_TILE + sub_y * TILE_SIZE + ty;
            int ow = blockIdx.x * OUTPUT_TILE + sub_x * TILE_SIZE + tx;
            
            if (oh < out_h && ow < out_w) {
                float max_val = -3.402823466e+38F;
                int start_r = (sub_y * TILE_SIZE + ty) * stride;
                int start_c = (sub_x * TILE_SIZE + tx) * stride;
                
                #pragma unroll
                for (int i = 0; i < k_size; ++i) {
                    #pragma unroll
                    for (int j = 0; j < k_size; ++j) {
                        float val = shared_mem[(start_r + i) * shared_dim + (start_c + j)];
                        if (val > max_val) max_val = val;
                    }
                }
                output[((bz * out_h + oh) * out_w + ow)] = max_val;
            }
        }
    }
}

void forward_cuda(const torch::Tensor& input, torch::Tensor& output, 
                  int k_size, int stride, int padding) {
    const int batch = input.size(0);
    const int channels = input.size(1);
    const int in_h = input.size(2);
    const int in_w = input.size(3);
    const int out_h = output.size(2);
    const int out_w = output.size(3);

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((out_w + (TILE_SIZE * MULT_FACTOR) - 1) / (TILE_SIZE * MULT_FACTOR),
              (out_h + (TILE_SIZE * MULT_FACTOR) - 1) / (TILE_SIZE * MULT_FACTOR),
              batch * channels);

    int shared_dim = (TILE_SIZE * MULT_FACTOR) * stride + k_size - 1;
    size_t shared_size = shared_dim * shared_dim * sizeof(float);

    max_pool2d_vec_kernel<<<grid, block, shared_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        batch, channels, in_h, in_w, out_h, out_w, 
        k_size, stride, padding);
}
"""

cpp_source = r"""
void forward_cuda(const torch::Tensor& input, torch::Tensor& output, int k_size, int stride, int padding);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_cuda, "Optimized MaxPool2D");
}
"""

module = load_inline(
    name='max_pool2d_optimized',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, maxpool_kernel_size, maxpool_stride, maxpool_padding, maxpool_dilation, maxpool_ceil_mode, maxpool_return_indices):
    if maxpool_dilation != 1 or maxpool_ceil_mode or maxpool_return_indices:
        raise NotImplementedError("Only basic MaxPool2d supported.")
    
    h_out = (x.shape[2] + 2 * maxpool_padding - maxpool_kernel_size) // maxpool_stride + 1
    w_out = (x.shape[3] + 2 * maxpool_padding - maxpool_kernel_size) // maxpool_stride + 1
    output = torch.empty((x.shape[0], x.shape[1], h_out, w_out), device=x.device, dtype=x.dtype)
    
    module.forward(x.contiguous(), output, maxpool_kernel_size, maxpool_stride, maxpool_padding)
    return output
