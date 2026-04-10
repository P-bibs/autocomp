# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_162105/code_4.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'output_padding', 'bias_shape', 'scaling_factor']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'bias', 'scaling_factor']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a transposed convolution, adds a bias term, clamps, scales, clamps, and divides.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor

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
    if 'scaling_factor' in flat_state:
        state_kwargs['scaling_factor'] = flat_state['scaling_factor']
    else:
        state_kwargs['scaling_factor'] = getattr(model, 'scaling_factor')
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

# The CUDA kernel uses a naive implementation of transpose convolution 
# to demonstrate the fusion of operations. In a production scenario, 
# one would use shared memory tiling or cuDNN-like strategies.

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_conv_transpose_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ add_bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int in_h, const int in_w,
    const int out_h, const int out_w,
    const int kernel_size,
    const int stride,
    const int padding,
    const float scale
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * out_h * out_w;
    
    if (out_idx >= total_elements) return;

    int tmp = out_idx;
    int w_out = tmp % out_w; tmp /= out_w;
    int h_out = tmp % out_h; tmp /= out_h;
    int c_out = tmp % out_channels; tmp /= out_channels;
    int b = tmp;

    float val = conv_bias[c_out] + add_bias[c_out];

    // Transpose convolution logic: sum over weights that contribute to this pixel
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                // Corresponding input source
                int hi = (h_out + padding - kh);
                int wi = (w_out + padding - kw);
                
                if (hi >= 0 && hi < in_h * stride && hi % stride == 0 &&
                    wi >= 0 && wi < in_w * stride && wi % stride == 0) {
                    
                    int h_idx = hi / stride;
                    int w_idx = wi / stride;
                    
                    int in_pos = ((b * in_channels + ic) * in_h + h_idx) * in_w + w_idx;
                    int w_pos = ((c_out * in_channels + ic) * kernel_size + kh) * kernel_size + kw;
                    
                    val += input[in_pos] * weight[w_pos];
                }
            }
        }
    }

    // Fused non-linearities: clamp(x, 0, 1) -> * scale -> clamp(x, 0, 1) -> / scale
    val = fmaxf(0.0f, fminf(1.0f, val));
    val = fmaxf(0.0f, fminf(1.0f, val * scale)) / scale;
    
    output[out_idx] = val;
}

void fused_conv_transpose_forward(
    torch::Tensor input, torch::Tensor weight, torch::Tensor conv_bias, 
    torch::Tensor add_bias, torch::Tensor output,
    int stride, int padding, float scale
) {
    int b = input.size(0), ic = input.size(1), ih = input.size(2), iw = input.size(3);
    int oc = weight.size(0), kh = weight.size(2), kw = weight.size(3);
    int oh = output.size(2), ow = output.size(3);

    int total = b * oc * oh * ow;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    fused_conv_transpose_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), conv_bias.data_ptr<float>(),
        add_bias.data_ptr<float>(), output.data_ptr<float>(),
        b, ic, oc, ih, iw, oh, ow, kh, stride, padding, scale
    );
}
"""

cpp_source = r"""
void fused_conv_transpose_forward(
    torch::Tensor input, torch::Tensor weight, torch::Tensor conv_bias, 
    torch::Tensor add_bias, torch::Tensor output,
    int stride, int padding, float scale);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_forward", &fused_conv_transpose_forward, "Fused Transpose Conv Op");
}
"""

fused_ext = load_inline(
    name='fused_ext', cpp_sources=cpp_source, cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True
)

def functional_model(x, *, conv_transpose_weight, conv_transpose_bias, conv_transpose_stride, 
                     conv_transpose_padding, conv_transpose_output_padding, conv_transpose_groups, 
                     conv_transpose_dilation, bias, scaling_factor):
    # Determine output shape
    b, ic, ih, iw = x.shape
    oc, _, kh, kw = conv_transpose_weight.shape
    oh = (ih - 1) * conv_transpose_stride - 2 * conv_transpose_padding + kh + conv_transpose_output_padding
    ow = (iw - 1) * conv_transpose_stride - 2 * conv_transpose_padding + kw + conv_transpose_output_padding
    
    output = torch.empty((b, oc, oh, ow), device=x.device, dtype=torch.float32)
    
    # Run custom kernel
    fused_ext.fused_forward(
        x.contiguous(), conv_transpose_weight.contiguous(), 
        conv_transpose_bias.flatten().contiguous(), bias.flatten().contiguous(), 
        output, conv_transpose_stride, conv_transpose_padding, scaling_factor
    )
    return output
