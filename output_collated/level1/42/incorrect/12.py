# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_055939/code_23.py
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

__global__ void max_pool2d_vectorized_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int in_h, int in_w, int out_h, int out_w,
    int k_size, int stride, int padding) {

    int n = blockIdx.z;
    // Map thread to output grid
    int col = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    int row = (blockIdx.y * blockDim.y + threadIdx.y) * 2;

    if (row >= out_h || col >= out_w) return;

    const float* input_ptr = input + (n * in_h * in_w);
    float max_vals[4] = {-1e30f, -1e30f, -1e30f, -1e30f};

    // Unroll pooling loop
    for (int ki = 0; ki < k_size; ++ki) {
        for (int kj = 0; kj < k_size; ++kj) {
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                int ih = (row + i) * stride + ki - padding;
                if (ih < 0 || ih >= in_h) continue;
                
                #pragma unroll
                for (int j = 0; j < 2; ++j) {
                    int iw = (col + j) * stride + kj - padding;
                    if (iw >= 0 && iw < in_w) {
                        float val = input_ptr[ih * in_w + iw];
                        int idx = i * 2 + j;
                        if (val > max_vals[idx]) max_vals[idx] = val;
                    }
                }
            }
        }
    }

    float* out_ptr = output + (n * out_h * out_w);
    if (row + 1 < out_h && col + 1 < out_w) {
        out_ptr[row * out_w + col] = max_vals[0];
        out_ptr[row * out_w + col + 1] = max_vals[1];
        out_ptr[(row + 1) * out_w + col] = max_vals[2];
        out_ptr[(row + 1) * out_w + col + 1] = max_vals[3];
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

    dim3 block(16, 16);
    dim3 grid((out_w/2 + block.x - 1) / block.x, (out_h/2 + block.y - 1) / block.y, batch * channels);

    max_pool2d_vectorized_kernel<<<grid, block>>>(
        input.contiguous().data_ptr<float>(), output.data_ptr<float>(),
        in_h, in_w, out_h, out_w, k_size, stride, padding
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void max_pool2d_forward(const torch::Tensor& input, torch::Tensor& output, int k_size, int stride, int padding);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_forward, "Optimized Vectorized Max Pool 2D");
}
"""

module = load_inline(
    name='max_pool2d_opt',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math', '-arch=sm_75'],
    with_cuda=True
)

def functional_model(x, *, maxpool_kernel_size, maxpool_stride, maxpool_padding, maxpool_dilation=1, maxpool_ceil_mode=False, maxpool_return_indices=False):
    h_in, w_in = x.shape[2], x.shape[3]
    h_out = (h_in + 2 * maxpool_padding - maxpool_kernel_size) // maxpool_stride + 1
    w_out = (w_in + 2 * maxpool_padding - maxpool_kernel_size) // maxpool_stride + 1
    
    output = torch.empty((x.shape[0], x.shape[1], h_out, w_out), device=x.device, dtype=x.dtype)
    module.forward(x, output, maxpool_kernel_size, maxpool_stride, maxpool_padding)
    return output
