# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_130013/code_18.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'output_padding', 'bias_shape']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'bias']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a 3D transposed convolution, followed by a sum, 
    a residual add, a multiplication, and another residual add.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
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
    # State for conv_transpose (nn.ConvTranspose3d)
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

# Optimized fused Conv3D Transpose + Bias + Post-processing CUDA kernel
# Using direct element-wise mapping for transposed convolution
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_conv3d_transpose_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ post_bias,
    float* __restrict__ output,
    int64_t batch_size,
    int64_t in_channels,
    int64_t out_channels,
    int64_t d_in, int64_t h_in, int64_t w_in,
    int64_t d_out, int64_t h_out, int64_t w_out,
    int64_t kd, int64_t kh, int64_t kw,
    int64_t sd, int64_t sh, int64_t sw,
    int64_t pd, int64_t ph, int64_t pw,
    int64_t groups
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = batch_size * out_channels * d_out * h_out * w_out;
    if (idx >= total) return;

    int64_t w_o = idx % w_out;
    int64_t h_o = (idx / w_out) % h_out;
    int64_t d_o = (idx / (w_out * h_out)) % d_out;
    int64_t oc = (idx / (w_out * h_out * d_out)) % out_channels;
    int64_t b = idx / (out_channels * w_out * h_out * d_out);

    float val = conv_bias[oc];
    int64_t group_size = out_channels / groups;
    int64_t g = oc / group_size;
    int64_t ic_start = g * (in_channels / groups);
    int64_t ic_end = ic_start + (in_channels / groups);

    for (int64_t ic = ic_start; ic < ic_end; ++ic) {
        for (int64_t i = 0; i < kd; ++i) {
            int64_t d_i = d_o - i + pd;
            if (d_i % sd != 0) continue;
            d_i /= sd;
            if (d_i < 0 || d_i >= d_in) continue;

            for (int64_t j = 0; j < kh; ++j) {
                int64_t h_i = h_o - j + ph;
                if (h_i % sh != 0) continue;
                h_i /= sh;
                if (h_i < 0 || h_i >= h_in) continue;

                for (int64_t k = 0; k < kw; ++k) {
                    int64_t w_i = w_o - k + pw;
                    if (w_i % sw != 0) continue;
                    w_i /= sw;
                    if (w_i < 0 || w_i >= w_in) continue;

                    int64_t in_idx = ((b * in_channels + ic) * d_in + d_i) * h_in * w_in + h_i * w_in + w_i;
                    int64_t wt_idx = ((oc * (in_channels / groups) + (ic - ic_start)) * kd + i) * kh * kw + j * kw + k;
                    val += input[in_idx] * weight[wt_idx];
                }
            }
        }
    }

    float b_val = post_bias[oc];
    output[idx] = ((val + b_val) + val) * val + val;
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_conv3d_transpose_kernel(const float* input, const float* weight, const float* conv_bias, const float* post_bias, float* output, 
                                   int64_t batch, int64_t ic, int64_t oc, int64_t di, int64_t hi, int64_t wi, 
                                   int64_t dout, int64_t hout, int64_t wout, int64_t kd, int64_t kh, int64_t kw, 
                                   int64_t sd, int64_t sh, int64_t sw, int64_t pd, int64_t ph, int64_t pw, int64_t groups);

torch::Tensor launch_fused(torch::Tensor input, torch::Tensor weight, torch::Tensor conv_bias, torch::Tensor post_bias,
                          int64_t sd, int64_t sh, int64_t sw, int64_t pd, int64_t ph, int64_t pw, 
                          int64_t opd, int64_t oph, int64_t opw, int64_t groups) {
    auto b = input.size(0); auto ic = input.size(1); auto di = input.size(2); auto hi = input.size(3); auto wi = input.size(4);
    auto oc = weight.size(1) * groups;
    auto kd = weight.size(2); auto kh = weight.size(3); auto kw = weight.size(4);
    auto dout = (di - 1) * sd - 2 * pd + kd + opd;
    auto hout = (hi - 1) * sh - 2 * ph + kh + oph;
    auto wout = (wi - 1) * sw - 2 * pw + kw + opw;
    auto output = torch::zeros({b, oc, dout, hout, wout}, input.options());
    
    int64_t total = b * oc * dout * hout * wout;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    
    fused_conv3d_transpose_kernel(input.data_ptr<float>(), weight.data_ptr<float>(), conv_bias.data_ptr<float>(), post_bias.data_ptr<float>(), output.data_ptr<float>(),
                                  b, ic, oc, di, hi, wi, dout, hout, wout, kd, kh, kw, sd, sh, sw, pd, ph, pw, groups);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_fused", &launch_fused);
}
"""

fused_ext = load_inline(name='fused_ext', cpp_sources=cpp_source, cuda_sources=cuda_kernel, extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True)

def functional_model(x, *, conv_transpose_weight, conv_transpose_bias, conv_transpose_stride, conv_transpose_padding, 
                     conv_transpose_output_padding, conv_transpose_groups, conv_transpose_dilation, bias):
    stride = conv_transpose_stride if isinstance(conv_transpose_stride, tuple) else (conv_transpose_stride,)*3
    pad = conv_transpose_padding if isinstance(conv_transpose_padding, tuple) else (conv_transpose_padding,)*3
    opad = conv_transpose_output_padding if isinstance(conv_transpose_output_padding, tuple) else (conv_transpose_output_padding,)*3
    return fused_ext.launch_fused(x.contiguous(), conv_transpose_weight.contiguous(), 
                                  conv_transpose_bias.flatten().contiguous(), bias.flatten().contiguous(),
                                  *stride, *pad, *opad, conv_transpose_groups)
