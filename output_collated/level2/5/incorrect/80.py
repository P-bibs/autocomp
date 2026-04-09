# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_123254/code_19.py
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

# The custom CUDA kernel implements the ConvTranspose2d operation manually 
# to achieve total fusion. Note: Manual implementation is used as per 
# instruction 6, replacing torch.nn.functional.conv_transpose2d.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_conv_transpose_tanh_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size, int in_channels, int out_channels,
    int in_h, int in_w, int out_h, int out_w,
    int kernel_size, int stride, int padding
) {
    int out_c = blockIdx.x;
    int b = blockIdx.y;
    int out_y = blockIdx.z / out_w;
    int out_x = blockIdx.z % out_w;

    float sum = 0.0f;
    // ConvTranspose2d implementation: iterating over input pixels that contribute to the output pixel
    for (int in_c = 0; in_c < in_channels; ++in_c) {
        for (int ky = 0; ky < kernel_size; ++ky) {
            for (int kx = 0; kx < kernel_size; ++kx) {
                int in_y = (out_y + padding - ky) / stride;
                int in_x = (out_x + padding - kx) / stride;

                if (in_y >= 0 && in_y < in_h && in_x >= 0 && in_x < in_w &&
                    (out_y + padding - ky) % stride == 0 && (out_x + padding - kx) % stride == 0) {
                    
                    int in_idx = ((b * in_channels + in_c) * in_h + in_y) * in_w + in_x;
                    int w_idx = ((out_c * in_channels + in_c) * kernel_size + ky) * kernel_size + kx;
                    sum += input[in_idx] * weight[w_idx];
                }
            }
        }
    }

    if (conv_bias != nullptr) sum += conv_bias[out_c];
    sum -= bias[out_c];
    
    int out_idx = ((b * out_channels + out_c) * out_h + out_y) * out_w + out_x;
    output[out_idx] = tanhf(sum);
}

void launch_fused_conv_transpose(
    const torch::Tensor input, const torch::Tensor weight,
    const torch::Tensor conv_bias, const torch::Tensor bias,
    torch::Tensor output, int stride, int padding
) {
    int b_size = input.size(0);
    int in_c = input.size(1);
    int in_h = input.size(2);
    int in_w = input.size(3);
    int out_c = output.size(1);
    int out_h = output.size(2);
    int out_w = output.size(3);
    int k = weight.size(2);

    dim3 grid(out_c, b_size, out_h * out_w);
    fused_conv_transpose_tanh_kernel<<<grid, 1>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(),
        conv_bias.defined() ? conv_bias.data_ptr<float>() : nullptr,
        bias.data_ptr<float>(), output.data_ptr<float>(),
        b_size, in_c, out_c, in_h, in_w, out_h, out_w, k, stride, padding
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void launch_fused_conv_transpose(const torch::Tensor input, const torch::Tensor weight, const torch::Tensor conv_bias, const torch::Tensor bias, torch::Tensor output, int stride, int padding);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_transpose", &launch_fused_conv_transpose, "Fused ConvTranspose2d Tanh");
}
"""

fused_ext = load_inline(name='fused_ext', cpp_sources=cpp_source, cuda_sources=cuda_kernel, extra_cuda_cflags=['-O3'])

def functional_model(x, *, conv_transpose_weight, conv_transpose_bias, conv_transpose_stride, 
                     conv_transpose_padding, conv_transpose_output_padding, conv_transpose_groups, 
                     conv_transpose_dilation, bias):
    # Calculate output shape manually based on parameters
    batch_size, in_channels, in_h, in_w = x.shape
    out_channels, _, k, _ = conv_transpose_weight.shape
    out_h = (in_h - 1) * conv_transpose_stride - 2 * conv_transpose_padding + k + conv_transpose_output_padding
    out_w = (in_w - 1) * conv_transpose_stride - 2 * conv_transpose_padding + k + conv_transpose_output_padding
    
    out = torch.empty((batch_size, out_channels, out_h, out_w), device=x.device)
    fused_ext.fused_conv_transpose(x, conv_transpose_weight, conv_transpose_bias, bias, out, conv_transpose_stride, conv_transpose_padding)
    return out
