# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_070107/code_4.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'bias']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv1d_weight', 'conv1d_bias', 'conv1d_stride', 'conv1d_padding', 'conv1d_dilation', 'conv1d_groups']
REQUIRED_FLAT_STATE_NAMES = ['conv1d_weight', 'conv1d_bias']


class ModelNew(nn.Module):
    """
    Performs a pointwise 2D convolution operation.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """

    def __init__(self, in_channels: int, out_channels: int, bias: bool=False):
        super(ModelNew, self).__init__()
        self.conv1d = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)

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
    # State for conv1d (nn.Conv2d)
    if 'conv1d_weight' in flat_state:
        state_kwargs['conv1d_weight'] = flat_state['conv1d_weight']
    else:
        state_kwargs['conv1d_weight'] = getattr(model.conv1d, 'weight', None)
    if 'conv1d_bias' in flat_state:
        state_kwargs['conv1d_bias'] = flat_state['conv1d_bias']
    else:
        state_kwargs['conv1d_bias'] = getattr(model.conv1d, 'bias', None)
    state_kwargs['conv1d_stride'] = model.conv1d.stride
    state_kwargs['conv1d_padding'] = model.conv1d.padding
    state_kwargs['conv1d_dilation'] = model.conv1d.dilation
    state_kwargs['conv1d_groups'] = model.conv1d.groups
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

# Optimized CUDA Kernel
# We use a 4D output mapping approach where each thread computes one output pixel 
# using a direct summation over the kernel spatial dimensions and input channels.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

template <typename scalar_t>
__global__ void conv2d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    int batch_size, int in_channels, int out_channels,
    int in_h, int in_w, int out_h, int out_w,
    int k_h, int k_w, int stride_h, int stride_w,
    int pad_h, int pad_w, int dil_h, int dil_w
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * out_channels * out_h * out_w;
    
    if (idx >= total) return;
    
    // Map linear index to 4D coordinates (N, C_out, H_out, W_out)
    int n = idx / (out_channels * out_h * out_w);
    int rem = idx % (out_channels * out_h * out_w);
    int c_out = rem / (out_h * out_w);
    rem = rem % (out_h * out_w);
    int h_out = rem / out_w;
    int w_out = rem % out_w;
    
    scalar_t sum = 0;
    int h_start = h_out * stride_h - pad_h;
    int w_start = w_out * stride_w - pad_w;
    
    // Compute cross-correlation
    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int kh = 0; kh < k_h; ++kh) {
            int h_in = h_start + kh * dil_h;
            if (h_in < 0 || h_in >= in_h) continue;
            for (int kw = 0; kw < k_w; ++kw) {
                int w_in = w_start + kw * dil_w;
                if (w_in >= 0 && w_in < in_w) {
                    scalar_t val = input[((n * in_channels + c_in) * in_h + h_in) * in_w + w_in];
                    scalar_t w = weight[((c_out * in_channels + c_in) * k_h + kh) * k_w + kw];
                    sum += val * w;
                }
            }
        }
    }
    
    if (bias != nullptr) {
        sum += bias[c_out];
    }
    
    output[idx] = sum;
}

void conv2d_forward_impl(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output,
    int s_h, int s_w, int p_h, int p_w, int d_h, int d_w
) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    int b = input.size(0); int ic = input.size(1);
    int ih = input.size(2); int iw = input.size(3);
    int oc = weight.size(0); int kh = weight.size(2); int kw = weight.size(3);
    int oh = output.size(2); int ow = output.size(3);
    
    int total_threads = b * oc * oh * ow;
    int threads_per_block = 256;
    int blocks = (total_threads + threads_per_block - 1) / threads_per_block;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv2d_forward", ([&] {
        conv2d_kernel<scalar_t><<<blocks, threads_per_block>>>(
            input.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(),
            bias.defined() ? bias.data_ptr<scalar_t>() : nullptr,
            output.data_ptr<scalar_t>(), b, ic, oc, ih, iw, oh, ow,
            kh, kw, s_h, s_w, p_h, p_w, d_h, d_w
        );
    }));
}
"""

cpp_source = r"""
void conv2d_forward_impl(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output,
    int s_h, int s_w, int p_h, int p_w, int d_h, int d_w);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv2d_forward", &conv2d_forward_impl, "Convolution forward");
}
"""

conv_ext = load_inline(
    name='optimized_conv2d',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, conv1d_weight, conv1d_bias, conv1d_stride, conv1d_padding, conv1d_dilation, conv1d_groups):
    # Standardize parameters for 2D
    s_h, s_w = (conv1d_stride, conv1d_stride) if isinstance(conv1d_stride, int) else conv1d_stride
    p_h, p_w = (conv1d_padding, conv1d_padding) if isinstance(conv1d_padding, int) else conv1d_padding
    d_h, d_w = (conv1d_dilation, conv1d_dilation) if isinstance(conv1d_dilation, int) else conv1d_dilation
    
    b, ic, ih, iw = x.shape
    oc, _, kh, kw = conv1d_weight.shape
    oh = (ih + 2 * p_h - d_h * (kh - 1) - 1) // s_h + 1
    ow = (iw + 2 * p_w - d_w * (kw - 1) - 1) // s_w + 1
    
    output = torch.empty(b, oc, oh, ow, device=x.device, dtype=x.dtype)
    conv_ext.conv2d_forward(x, conv1d_weight, conv1d_bias, output, s_h, s_w, p_h, p_w, d_h, d_w)
    return output
