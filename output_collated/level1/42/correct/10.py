# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_052448/code_14.py
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

# CUDA kernel implementation
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <float.h>

__global__ void max_pool2d_forward_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    long* __restrict__ indices,
    int batch_size,
    int channels,
    int in_h,
    int in_w,
    int out_h,
    int out_w,
    int k_h,
    int k_w,
    int s_h,
    int s_w,
    int p_h,
    int p_w,
    int d_h,
    int d_w,
    bool return_indices
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * out_h * out_w;
    
    if (idx >= total_elements) return;
    
    int w_out = idx % out_w;
    int h_out = (idx / out_w) % out_h;
    int c = (idx / (out_w * out_h)) % channels;
    int n = idx / (out_w * out_h * channels);
    
    int h_start = h_out * s_h - p_h;
    int w_start = w_out * s_w - p_w;
    
    float max_val = -FLT_MAX;
    long max_idx = -1;
    
    int base_input = (n * channels + c) * in_h * in_w;
    
    for (int kh = 0; kh < k_h; ++kh) {
        int h_in = h_start + kh * d_h;
        if (h_in >= 0 && h_in < in_h) {
            for (int kw = 0; kw < k_w; ++kw) {
                int w_in = w_start + kw * d_w;
                if (w_in >= 0 && w_in < in_w) {
                    int offset = base_input + h_in * in_w + w_in;
                    float val = input[offset];
                    if (val > max_val) {
                        max_val = val;
                        max_idx = (long)offset;
                    }
                }
            }
        }
    }
    
    output[idx] = max_val;
    if (return_indices) {
        indices[idx] = max_idx;
    }
}

void max_pool2d_forward(
    const at::Tensor& input,
    at::Tensor& output,
    at::Tensor& indices,
    int k_h, int k_w,
    int s_h, int s_w,
    int p_h, int p_w,
    int d_h, int d_w,
    bool return_indices
) {
    int batch = input.size(0);
    int channels = input.size(1);
    int in_h = input.size(2);
    int in_w = input.size(3);
    int out_h = output.size(2);
    int out_w = output.size(3);
    
    int total_elements = batch * channels * out_h * out_w;
    int threads_per_block = 256;
    int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    max_pool2d_forward_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        return_indices ? indices.data_ptr<long>() : nullptr,
        batch, channels, in_h, in_w, out_h, out_w,
        k_h, k_w, s_h, s_w, p_h, p_w, d_h, d_w,
        return_indices
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void max_pool2d_forward(
    const at::Tensor& input,
    at::Tensor& output,
    at::Tensor& indices,
    int k_h, int k_w,
    int s_h, int s_w,
    int p_h, int p_w,
    int d_h, int d_w,
    bool return_indices
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("max_pool2d_forward", &max_pool2d_forward, "Max Pool 2D CUDA");
}
"""

maxpool_ext = load_inline(
    name='maxpool_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(
    x,
    *,
    maxpool_kernel_size,
    maxpool_stride,
    maxpool_padding,
    maxpool_dilation,
    maxpool_ceil_mode,
    maxpool_return_indices,
):
    h, w = x.shape[2], x.shape[3]
    if maxpool_ceil_mode:
        oh = int(((h + 2 * maxpool_padding - maxpool_dilation * (maxpool_kernel_size - 1) - 1) + maxpool_stride - 1) / maxpool_stride + 1)
        ow = int(((w + 2 * maxpool_padding - maxpool_dilation * (maxpool_kernel_size - 1) - 1) + maxpool_stride - 1) / maxpool_stride + 1)
    else:
        oh = (h + 2 * maxpool_padding - maxpool_dilation * (maxpool_kernel_size - 1) - 1) // maxpool_stride + 1
        ow = (w + 2 * maxpool_padding - maxpool_dilation * (maxpool_kernel_size - 1) - 1) // maxpool_stride + 1
    
    output = torch.empty((x.shape[0], x.shape[1], oh, ow), device=x.device, dtype=x.dtype)
    indices = torch.empty_like(output, dtype=torch.long) if maxpool_return_indices else torch.tensor([], device=x.device)
    
    maxpool_ext.max_pool2d_forward(
        x, output, indices,
        maxpool_kernel_size, maxpool_kernel_size,
        maxpool_stride, maxpool_stride,
        maxpool_padding, maxpool_padding,
        maxpool_dilation, maxpool_dilation,
        maxpool_return_indices
    )
    return (output, indices) if maxpool_return_indices else output

batch_size = 32
channels = 64
height = 512
width = 512
kernel_size = 4
stride = 1
padding = 1
dilation = 1

def get_init_inputs():
    return [kernel_size, stride, padding, dilation]

def get_inputs():
    return [torch.rand(batch_size, channels, height, width, device='cuda')]
