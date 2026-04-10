# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_055939/code_27.py
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

# The kernel uses shared memory tiling to cache input patches.
# Each block computes a 16x16 tile of the output.
# Total shared memory size used is (16 + k_size - 1)^2 * sizeof(float).
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <float.h>

__global__ void max_pool2d_kernel_shared(
    const float* __restrict__ input,
    float* __restrict__ output,
    int in_h, int in_w,
    int out_h, int out_w,
    int k_size, int stride, int padding) {

    extern __shared__ float shared_mem[];
    
    const int TILE_SIZE = 16;
    // Each thread block processes a TILE_SIZE x TILE_SIZE output block
    // We need (TILE_SIZE-1)*stride + k_size input elements to cover the area.
    const int input_tile_dim = (TILE_SIZE - 1) * stride + k_size;
    
    int block_out_h = blockIdx.y * TILE_SIZE;
    int block_out_w = blockIdx.x * TILE_SIZE;
    int bz = blockIdx.z;
    
    const float* batch_input = input + (bz * in_h * in_w);
    float* batch_output = output + (bz * out_h * out_w);
    
    int start_h = block_out_h * stride - padding;
    int start_w = block_out_w * stride - padding;
    
    // Load input patch into shared memory cooperatively
    for (int idx = threadIdx.y * blockDim.x + threadIdx.x; 
         idx < input_tile_dim * input_tile_dim; 
         idx += blockDim.x * blockDim.y) {
        int li = idx / input_tile_dim;
        int lj = idx % input_tile_dim;
        int gi = start_h + li;
        int gj = start_w + lj;
        
        if (gi >= 0 && gi < in_h && gj >= 0 && gj < in_w) {
            shared_mem[idx] = batch_input[gi * in_w + gj];
        } else {
            shared_mem[idx] = -FLT_MAX;
        }
    }
    __syncthreads();
    
    int oh = block_out_h + threadIdx.y;
    int ow = block_out_w + threadIdx.x;
    
    if (oh < out_h && ow < out_w) {
        float max_val = -FLT_MAX;
        int base_ih = threadIdx.y * stride;
        int base_iw = threadIdx.x * stride;
        
        #pragma unroll
        for (int ki = 0; ki < k_size; ++ki) {
            #pragma unroll
            for (int kj = 0; kj < k_size; ++kj) {
                int sh_idx = (base_ih + ki) * input_tile_dim + (base_iw + kj);
                float val = shared_mem[sh_idx];
                if (val > max_val) max_val = val;
            }
        }
        batch_output[oh * out_w + ow] = max_val;
    }
}

void max_pool2d_forward_cuda(const torch::Tensor& input, torch::Tensor& output, 
                             int k_size, int stride, int padding) {
    const int batch = input.size(0);
    const int channels = input.size(1);
    const int in_h = input.size(2);
    const int in_w = input.size(3);
    const int out_h = output.size(2);
    const int out_w = output.size(3);
    
    const int TILE_SIZE = 16;
    const int input_tile_dim = (TILE_SIZE - 1) * stride + k_size;
    
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((out_w + TILE_SIZE - 1) / TILE_SIZE, 
              (out_h + TILE_SIZE - 1) / TILE_SIZE, 
              batch * channels);
    
    size_t shared_mem_size = input_tile_dim * input_tile_dim * sizeof(float);
    
    max_pool2d_kernel_shared<<<grid, block, shared_mem_size>>>(
        input.contiguous().data_ptr<float>(), 
        output.data_ptr<float>(),
        in_h, in_w, out_h, out_w,
        k_size, stride, padding
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void max_pool2d_forward_cuda(const torch::Tensor& input, torch::Tensor& output, int k_size, int stride, int padding);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_forward_cuda, "Optimized shared memory Max Pool 2D");
}
"""

# Compile the extension
maxpool_module = load_inline(
    name='maxpool_ext',
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
    maxpool_module.forward(x, output, maxpool_kernel_size, maxpool_stride, maxpool_padding)
    return output
