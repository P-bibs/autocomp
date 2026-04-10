# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_053509/code_20.py
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

__global__ void max_pool2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int channels, int in_h, int in_w,
    int out_h, int out_w,
    int batch_size_channels,
    int k_size, int stride, int padding) {

    // Grid: (out_w/2, out_h/2, batch * channels)
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z; 

    int ow = (bx * blockDim.x + threadIdx.x) * 2;
    int oh = (by * blockDim.y + threadIdx.y) * 2;

    float max_vals[2][2];
    #pragma unroll
    for(int i=0; i<2; ++i) {
        #pragma unroll
        for(int j=0; j<2; ++j) {
            max_vals[i][j] = -FLT_MAX;
        }
    }

    const float* input_ptr = input + (bz * in_h * in_w);

    // Compute pooling
    #pragma unroll
    for (int ki = 0; ki < k_size; ++ki) {
        #pragma unroll
        for (int kj = 0; kj < k_size; ++kj) {
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                #pragma unroll
                for (int j = 0; j < 2; ++j) {
                    int ih = (oh + i) * stride + ki - padding;
                    int iw = (ow + j) * stride + kj - padding;

                    if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                        float val = input_ptr[ih * in_w + iw];
                        if (val > max_vals[i][j]) max_vals[i][j] = val;
                    }
                }
            }
        }
    }

    // Write back to Global Memory
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            if ((oh + i) < out_h && (ow + j) < out_w) {
                output[(bz * out_h + (oh + i)) * out_w + (ow + j)] = max_vals[i][j];
            }
        }
    }
}

void max_pool2d_forward(const torch::Tensor& input, torch::Tensor& output, 
                        int k_size, int stride, int padding) {
    const int batch = input.size(0);
    const int channels = input.size(1);
    const int in_h = input.size(2);
    const int in_w = input.size(3);
    const int out_h = output.size(2);
    const int out_w = output.size(3);

    // Block size 16x16: 256 threads, each processes 4 output elements
    dim3 block(16, 16);
    dim3 grid((out_w/2 + block.x - 1) / block.x, (out_h/2 + block.y - 1) / block.y, batch * channels);

    max_pool2d_kernel<<<grid, block>>>(
        input.contiguous().data_ptr<float>(), output.data_ptr<float>(),
        channels, in_h, in_w, out_h, out_w, batch * channels,
        k_size, stride, padding
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void max_pool2d_forward(const torch::Tensor& input, torch::Tensor& output, int k_size, int stride, int padding);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_forward, "Optimized register-tiled Max Pool 2D");
}
"""

module = load_inline(
    name='max_pool2d_opt',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math', '-arch=sm_75'],
    with_cuda=True
)

def functional_model(x, *, maxpool_kernel_size, maxpool_stride, maxpool_padding, maxpool_dilation, maxpool_ceil_mode, maxpool_return_indices):
    h_in, w_in = x.shape[2], x.shape[3]
    h_out = (h_in + 2 * maxpool_padding - maxpool_kernel_size) // maxpool_stride + 1
    w_out = (w_in + 2 * maxpool_padding - maxpool_kernel_size) // maxpool_stride + 1
    
    output = torch.empty((x.shape[0], x.shape[1], h_out, w_out), device=x.device, dtype=x.dtype)
    module.forward(x, output, maxpool_kernel_size, maxpool_stride, maxpool_padding)
    return output
