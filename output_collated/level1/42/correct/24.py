# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_053509/code_17.py
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

# --- Optimized CUDA Kernel ---
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// MaxPool2D kernel optimized for coalesced global memory access.
// We use a 2D block structure (16x16) to ensure that neighboring threads
// in the warp access adjacent memory, maximizing L1 cache line utilization.
__global__ void max_pool2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size, int channels,
    int in_h, int in_w,
    int out_h, int out_w,
    int k_size, int stride, int padding) {

    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int bz = blockIdx.z;
    
    if (ow >= out_w || oh >= out_h) return;

    int c = bz % channels;
    int b = bz / channels;

    int ih_start = oh * stride - padding;
    int iw_start = ow * stride - padding;

    float max_val = -3.40282e+38f; // FLT_MIN

    // Using pointers to navigate the input tensor efficiently
    const float* input_ptr = input + (b * channels + c) * (in_h * in_w);

    #pragma unroll
    for (int i = 0; i < k_size; ++i) {
        int ih = ih_start + i;
        if (ih >= 0 && ih < in_h) {
            const float* row_ptr = input_ptr + ih * in_w;
            #pragma unroll
            for (int j = 0; j < k_size; ++j) {
                int iw = iw_start + j;
                if (iw >= 0 && iw < in_w) {
                    float val = row_ptr[iw];
                    if (val > max_val) max_val = val;
                }
            }
        }
    }

    output[((b * channels + c) * out_h + oh) * out_w + ow] = max_val;
}

void max_pool2d_forward(const torch::Tensor& input, torch::Tensor& output, 
                        int k_size, int stride, int padding) {
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int in_h = input.size(2);
    const int in_w = input.size(3);
    const int out_h = output.size(2);
    const int out_w = output.size(3);

    // Using 16x16 threads per block is optimal for 2D spatial data
    dim3 threads(16, 16);
    dim3 blocks(
        (out_w + threads.x - 1) / threads.x,
        (out_h + threads.y - 1) / threads.y,
        batch_size * channels
    );

    max_pool2d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        batch_size, channels, in_h, in_w, out_h, out_w,
        k_size, stride, padding
    );
}
"""

# --- C++ Binding ---
cpp_source = r"""
#include <torch/extension.h>

void max_pool2d_forward(const torch::Tensor& input, torch::Tensor& output, 
                        int k_size, int stride, int padding);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_forward, "Optimized Max Pool 2D Forward");
}
"""

# Compile the extension
module = load_inline(
    name='max_pool2d_optimized',
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
    module.forward(x.contiguous(), output, maxpool_kernel_size, maxpool_stride, maxpool_padding)
    
    if maxpool_return_indices:
        raise NotImplementedError("Indices are not supported in custom optimized kernel.")
    return output
