# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_115905/code_12.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'bias_shape', 'stride', 'padding', 'output_padding']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'bias']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a transposed convolution, subtracts a bias term, and applies tanh activation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, bias_shape, stride=2, padding=1, output_padding=1):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))

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
    # State for conv_transpose (nn.ConvTranspose2d)
    if 'conv_transpose_weight' in flat_state:
        state_kwargs['conv_transpose_weight'] = flat_state['conv_transpose_weight']
    else:
        state_kwargs['conv_transpose_weight'] = getattr(model.conv_transpose, 'weight', None)
    if 'conv_transpose_bias' in flat_state:
        state_kwargs['conv_transpose_bias'] = flat_state['conv_transpose_bias']
    else:
        state_kwargs['conv_transpose_bias'] = getattr(model.conv_transpose, 'bias', None)
    state_kwargs['conv_transpose_stride'] = model.conv_transpose.stride
    state_kwargs['conv_transpose_padding'] = model.conv_transpose.padding
    state_kwargs['conv_transpose_output_padding'] = model.conv_transpose.output_padding
    state_kwargs['conv_transpose_groups'] = model.conv_transpose.groups
    state_kwargs['conv_transpose_dilation'] = model.conv_transpose.dilation
    if 'bias' in flat_state:
        state_kwargs['bias'] = flat_state['bias']
    else:
        state_kwargs['bias'] = getattr(model, 'bias')
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

# CUDA kernel providing combined Transposed Convolution + Bias Addition + Tanh
# This minimizes global memory round-trips by fusing the operations.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose_bias_tanh_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int in_c, int in_h, int in_w,
    int out_c, int out_h, int out_w,
    int kernel_size, int stride, int padding) {

    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * out_c * out_h * out_w;

    if (gid >= total_elements) return;

    // Decoding linear index to coordinate
    int temp = gid;
    int w_out = temp % out_w; temp /= out_w;
    int h_out = temp % out_h; temp /= out_h;
    int c_out = temp % out_c; temp /= out_c;
    int n = temp;

    float sum = 0.0f;

    // Transposed Conv Logic: 
    // An output pixel at (h_out, w_out) is influenced by an input window.
    // Equivalent to sliding a kernel over the input with padding/striding.
    for (int ic = 0; ic < in_c; ++ic) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                // Input coordinate mapping
                int h_in_f = h_out + padding - kh;
                int w_in_f = w_out + padding - kw;

                if (h_in_f % stride == 0 && w_in_f % stride == 0) {
                    int h_in = h_in_f / stride;
                    int w_in = w_in_f / stride;

                    if (h_in >= 0 && h_in < in_h && w_in >= 0 && w_in < in_w) {
                        int i_idx = ((n * in_c + ic) * in_h + h_in) * in_w + w_in;
                        int w_idx = ((c_out * in_c + ic) * kernel_size + kh) * kernel_size + kw;
                        sum += input[i_idx] * weight[w_idx];
                    }
                }
            }
        }
    }

    // Apply bias and Tanh
    sum += bias[c_out];
    output[gid] = tanhf(sum);
}
"""

cpp_source = r"""
#include <torch/extension.h>

void launch_conv_transpose_bias_tanh(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output,
    int stride, int padding, int kernel_size);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &launch_conv_transpose_bias_tanh, "Fused ConvTranspose2d+Bias+Tanh");
}
"""

# The kernel launch configuration is handled in C++ here
cuda_source_wrapper = cuda_kernel + r"""
void launch_conv_transpose_bias_tanh(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output,
    int stride, int padding, int kernel_size) {
    
    int N = input.size(0), in_c = input.size(1), in_h = input.size(2), in_w = input.size(3);
    int out_c = weight.size(0), out_h = output.size(2), out_w = output.size(3);
    
    int total_elements = N * out_c * out_h * out_w;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    
    conv_transpose_bias_tanh_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), N, in_c, in_h, in_w, out_c, out_h, out_w,
        kernel_size, stride, padding
    );
}
"""

fused_lib = load_inline(
    name='fused_conv_ops',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source_wrapper,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(
    x,
    *,
    conv_transpose_weight,
    conv_transpose_bias,
    conv_transpose_stride,
    conv_transpose_padding,
    conv_transpose_output_padding,
    conv_transpose_groups,
    conv_transpose_dilation,
):
    # Output dim calc
    n, ic, ih, iw = x.shape
    oc, _, ks, _ = conv_transpose_weight.shape
    
    oh = (ih - 1) * conv_transpose_stride - 2 * conv_transpose_padding + ks + conv_transpose_output_padding
    ow = (iw - 1) * conv_transpose_stride - 2 * conv_transpose_padding + ks + conv_transpose_output_padding
    
    output = torch.empty((n, oc, oh, ow), device=x.device, dtype=x.dtype)
    
    # Run custom fused CUDA kernel
    fused_lib.forward(
        x.contiguous(),
        conv_transpose_weight.contiguous(),
        conv_transpose_bias.view(-1).contiguous(),
        output,
        conv_transpose_stride,
        conv_transpose_padding,
        ks
    )
    
    return output
