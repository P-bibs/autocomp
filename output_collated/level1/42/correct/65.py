# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_055939/code_22.py
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
#include <algorithm>

__global__ void max_pool2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int N, const int C, const int H, const int W,
    const int out_h, const int out_w,
    const int k_size, const int stride, const int padding) {

    const int total_elements = N * C * out_h * out_w;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= total_elements) return;

    // Coordinate mapping
    int rem = tid;
    const int ow = rem % out_w; rem /= out_w;
    const int oh = rem % out_h; rem /= out_h;
    const int c = rem % C; rem /= C;
    const int n = rem;

    const int ih_start = oh * stride - padding;
    const int iw_start = ow * stride - padding;

    float max_val = -1e38f;

    // Direct global memory access with compiler-assisted unrolling
    // The compiler can better pipeline these loads than manual shared memory management
    #pragma unroll
    for (int di = 0; di < k_size; ++di) {
        int ih = ih_start + di;
        if (ih >= 0 && ih < H) {
            #pragma unroll
            for (int dj = 0; dj < k_size; ++dj) {
                int iw = iw_start + dj;
                if (iw >= 0 && iw < W) {
                    float val = input[((n * C + c) * H + ih) * W + iw];
                    if (val > max_val) max_val = val;
                }
            }
        }
    }
    output[tid] = max_val;
}

void max_pool2d_forward_cuda(const torch::Tensor& input, torch::Tensor& output,
                             int k_size, int stride, int padding) {
    const int N = input.size(0);
    const int C = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);
    const int out_h = output.size(2);
    const int out_w = output.size(3);
    
    int total_elements = output.numel();
    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;

    max_pool2d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        N, C, H, W, out_h, out_w, k_size, stride, padding
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void max_pool2d_forward_cuda(const torch::Tensor& input, torch::Tensor& output,
                             int k_size, int stride, int padding);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_forward_cuda, "Max pool 2D forward optimized");
}
"""

# Compile the optimized kernel
optimized_ext = load_inline(
    name='max_pool2d_opt',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math', '-arch=sm_75'],
    with_cuda=True
)

def functional_model(x, *, maxpool_kernel_size, maxpool_stride, maxpool_padding,
                     maxpool_dilation, maxpool_ceil_mode, maxpool_return_indices):
    h_in, w_in = x.shape[2], x.shape[3]
    h_out = (h_in + 2 * maxpool_padding - maxpool_kernel_size) // maxpool_stride + 1
    w_out = (w_in + 2 * maxpool_padding - maxpool_kernel_size) // maxpool_stride + 1
    
    output = torch.empty((x.shape[0], x.shape[1], h_out, w_out), 
                         device=x.device, dtype=x.dtype)
    
    optimized_ext.forward(x.contiguous(), output, 
                          maxpool_kernel_size, maxpool_stride, maxpool_padding)
    
    return output
