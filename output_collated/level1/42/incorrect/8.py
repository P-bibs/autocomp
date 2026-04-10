# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_054839/code_21.py
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

__global__ void max_pool2d_vectorized_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int in_h, int in_w,
    int out_h, int out_w, int batch_channels,
    int k_size, int stride, int padding) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Each thread handles a 2x2 block which corresponds to 4 output pixels
    if (tid >= batch_channels * out_h * (out_w / 2)) return;

    int bc = tid / (out_h * (out_w / 2));
    int rem = tid % (out_h * (out_w / 2));
    int oh = rem / (out_w / 2);
    int ow = (rem % (out_w / 2)) * 2;

    const float* in_ptr = input + (bc * in_h * in_w);
    float* out_ptr = output + (bc * out_h * out_w);

    float max00 = -1e30f, max01 = -1e30f, max10 = -1e30f, max11 = -1e30f;

    // Kernel traversal with localized access
    #pragma unroll
    for (int ki = 0; ki < k_size; ++ki) {
        int ih0 = oh * stride + ki - padding;
        int ih1 = (oh + 1) * stride + ki - padding;
        
        #pragma unroll
        for (int kj = 0; kj < k_size; ++kj) {
            int iw0 = ow * stride + kj - padding;
            int iw1 = (ow + 1) * stride + kj - padding;

            if (ih0 >= 0 && ih0 < in_h) {
                if (iw0 >= 0 && iw0 < in_w) max00 = fmaxf(max00, in_ptr[ih0 * in_w + iw0]);
                if (iw1 >= 0 && iw1 < in_w) max01 = fmaxf(max01, in_ptr[ih0 * in_w + iw1]);
            }
            if (ih1 >= 0 && ih1 < in_h) {
                if (iw0 >= 0 && iw0 < in_w) max10 = fmaxf(max10, in_ptr[ih1 * in_w + iw0]);
                if (iw1 >= 0 && iw1 < in_w) max11 = fmaxf(max11, in_ptr[ih1 * in_w + iw1]);
            }
        }
    }
    
    // Coalesced write
    out_ptr[oh * out_w + ow] = max00;
    out_ptr[oh * out_w + (ow + 1)] = max01;
    out_ptr[(oh + 1) * out_w + ow] = max10;
    out_ptr[(oh + 1) * out_w + (ow + 1)] = max11;
}

void max_pool2d_forward(torch::Tensor input, torch::Tensor output, int k, int s, int p) {
    int b = input.size(0), c = input.size(1), ih = input.size(2), iw = input.size(3);
    int oh = output.size(2), ow = output.size(3);
    int total_threads = b * c * oh * (ow / 2);
    int block_size = 256;
    int grid_size = (total_threads + block_size - 1) / block_size;
    
    max_pool2d_vectorized_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), ih, iw, oh, ow, b*c, k, s, p);
}
"""

cpp_source = r"""
void max_pool2d_forward(torch::Tensor input, torch::Tensor output, int k, int s, int p);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool2d_forward, "Optimized Max Pool 2D");
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
    # Ensure input is contiguous for optimal loading
    module.forward(x.contiguous(), output, maxpool_kernel_size, maxpool_stride, maxpool_padding)
    return output
