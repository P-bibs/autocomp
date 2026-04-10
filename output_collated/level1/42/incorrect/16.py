# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_061245/code_27.py
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

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector_types.h>

#define TILE_SIZE 8

__global__ void max_pool2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int in_h, const int in_w,
    const int out_h, const int out_w,
    const int k_size, const int stride, const int padding) {

    extern __shared__ float shared_input[];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int bz = blockIdx.z;

    const int shared_width = TILE_SIZE * stride + k_size - 1;
    const int shared_height = TILE_SIZE * stride + k_size - 1;
    const int shared_width_padded = shared_width + 1; 

    const int ih_start = by * TILE_SIZE * stride - padding;
    const int iw_start = bx * TILE_SIZE * stride - padding;

    // Use float4 pointer for coalesced loading
    const int total_elements = shared_width * shared_height;
    float4* shared_input_v = reinterpret_cast<float4*>(shared_input);
    const float* input_ptr = input + (bz * in_h * in_w);

    // Load data using float4 vectors where possible
    for (int i = (ty * blockDim.x + tx); i < (total_elements / 4); i += (blockDim.x * blockDim.y)) {
        int idx = i * 4;
        int row = idx / shared_width;
        int col = idx % shared_width;
        
        float4 v;
        #pragma unroll
        for(int k=0; k<4; ++k) {
            int cur_r = row + (col + k) / shared_width;
            int cur_c = (col + k) % shared_width;
            int ih = ih_start + cur_r;
            int iw = iw_start + cur_c;
            float val = -1e38f;
            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                val = input_ptr[ih * in_w + iw];
            }
            if (k==0) v.x = val; else if (k==1) v.y = val; else if (k==2) v.z = val; else v.w = val;
        }
        shared_input_v[i] = v;
    }

    // Handle remainder
    for (int i = (total_elements / 4) * 4 + (ty * blockDim.x + tx); i < total_elements; i += (blockDim.x * blockDim.y)) {
        int r = i / shared_width;
        int c = i % shared_width;
        int ih = ih_start + r;
        int iw = iw_start + c;
        float val = -1e38f;
        if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
            val = input_ptr[ih * in_w + iw];
        }
        shared_input[r * shared_width_padded + c] = val;
    }

    __syncthreads();

    int ow = bx * TILE_SIZE + tx;
    int oh = by * TILE_SIZE + ty;

    if (ow < out_w && oh < out_h) {
        float max_val = -1e38f;
        for (int di = 0; di < k_size; ++di) {
            int sh_i = ty * stride + di;
            for (int dj = 0; dj < k_size; ++dj) {
                int sh_j = tx * stride + dj;
                max_val = fmaxf(max_val, shared_input[sh_i * shared_width_padded + sh_j]);
            }
        }
        output[(bz * out_h + oh) * out_w + ow] = max_val;
    }
}

void max_pool2d_forward(const torch::Tensor& input, torch::Tensor& output, 
                        const int k_size, const int stride, const int padding) {
    const int batch = input.size(0);
    const int channels = input.size(1);
    const int in_h = input.size(2);
    const int in_w = input.size(3);
    const int out_h = output.size(2);
    const int out_w = output.size(3);

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((out_w + TILE_SIZE - 1) / TILE_SIZE, (out_h + TILE_SIZE - 1) / TILE_SIZE, batch * channels);
    
    int shared_width = TILE_SIZE * stride + k_size - 1;
    int shared_height = TILE_SIZE * stride + k_size - 1;
    size_t sh_mem = (shared_width + 1) * shared_height * sizeof(float);

    max_pool2d_kernel<<<grid, block, sh_mem>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        in_h, in_w, out_h, out_w, k_size, stride, padding);
}
"""

cpp_source = r"""
#include <torch/extension.h>
void max_pool2d_forward(const torch::Tensor& input, torch::Tensor& output, const int k, const int s, const int p);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_forward, "Max pool 2D forward");
}
"""

module = load_inline(name='max_pool2d_optimized', cpp_sources=cpp_source, cuda_sources=cuda_kernel, 
                     extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True)

def functional_model(x, *, maxpool_kernel_size, maxpool_stride, maxpool_padding,
                     maxpool_dilation, maxpool_ceil_mode, maxpool_return_indices):
    h_in, w_in = x.shape[2], x.shape[3]
    h_out = (h_in + 2 * maxpool_padding - maxpool_kernel_size) // maxpool_stride + 1
    w_out = (w_in + 2 * maxpool_padding - maxpool_kernel_size) // maxpool_stride + 1
    output = torch.empty((x.shape[0], x.shape[1], h_out, w_out), device=x.device, dtype=x.dtype)
    module.forward(x.contiguous(), output, maxpool_kernel_size, maxpool_stride, maxpool_padding)
    return output
