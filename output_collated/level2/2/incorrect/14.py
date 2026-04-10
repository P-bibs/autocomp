# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_163027/code_7.py
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

# 1. Custom CUDA Implementation
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Fused Transposed Convolution (Naive but functional) + Bias Add + Clamp
// This replaces F.conv_transpose2d + Pointwise chain
__global__ void fused_conv_transpose_bias_clamp_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int in_C, int out_C, int H, int W,
    int kH, int kW, int stride, int padding, 
    float max_clamp) 
{
    int out_H = (H - 1) * stride + kH - 2 * padding;
    int out_W = (W - 1) * stride + kW - 2 * padding;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * out_C * out_H * out_W;

    if (idx < total) {
        int n = idx / (out_C * out_H * out_W);
        int rem = idx % (out_C * out_H * out_W);
        int oc = rem / (out_H * out_W);
        int pos = rem % (out_H * out_W);
        int oh = pos / out_W;
        int ow = pos % out_W;

        float val = bias[oc];
        for (int ic = 0; ic < in_C; ++ic) {
            for (int kh = 0; kh < kH; ++kh) {
                for (int kw = 0; kw < kW; ++kw) {
                    int ih = (oh + padding - kh);
                    int iw = (ow + padding - kw);
                    if (ih % stride == 0 && iw % stride == 0) {
                        ih /= stride; iw /= stride;
                        if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                            val += input[((n * in_C + ic) * H + ih) * W + iw] * 
                                   weight[((oc * in_C + ic) * kH + kh) * kW + kw];
                        }
                    }
                }
            }
        }
        if (val < 0.0f) val = 0.0f;
        else if (val > max_clamp) val = max_clamp;
        output[idx] = val;
    }
}

void fused_op(const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias, at::Tensor& output, 
              int stride, int padding, float max_clamp) {
    int N = input.size(0); int in_C = input.size(1);
    int H = input.size(2); int W = input.size(3);
    int out_C = weight.size(1);
    int kH = weight.size(2); int kW = weight.size(3);
    int out_H = (H - 1) * stride + kH - 2 * padding;
    int out_W = (W - 1) * stride + kW - 2 * padding;
    
    int total = N * out_C * out_H * out_W;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    fused_conv_transpose_bias_clamp_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), N, in_C, out_C, H, W, kH, kW, stride, padding, max_clamp);
}
"""

cpp_source = r"""
void fused_op(const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias, at::Tensor& output, 
              int stride, int padding, float max_clamp);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op, "Fused ConvTranspose + Bias + Clamp");
}
"""

module = load_inline(name='fused_conv', cpp_sources=cpp_source, cuda_sources=cuda_source, with_cuda=True)

def functional_model(x, *, conv_transpose_weight, conv_transpose_bias, conv_transpose_stride, 
                     conv_transpose_padding, conv_transpose_output_padding, conv_transpose_groups, 
                     conv_transpose_dilation, bias, scaling_factor):
    max_clamp = 1.0 / scaling_factor
    n, in_c, h, w = x.shape
    out_c = conv_transpose_weight.size(1) # Transposed weight shape: (in_c, out_c, k, k)
    out_h = (h - 1) * conv_transpose_stride + conv_transpose_weight.size(2) - 2 * conv_transpose_padding
    out_w = (w - 1) * conv_transpose_stride + conv_transpose_weight.size(3) - 2 * conv_transpose_padding
    
    out = torch.empty((n, out_c, out_h, out_w), device=x.device)
    module.fused_op(x, conv_transpose_weight, bias.view(-1), out, conv_transpose_stride, conv_transpose_padding, max_clamp)
    return out
