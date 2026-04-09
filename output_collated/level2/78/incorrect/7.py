# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_030214/code_7.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'max_pool1_kernel_size', 'max_pool1_stride', 'max_pool1_padding', 'max_pool1_dilation', 'max_pool1_ceil_mode', 'max_pool1_return_indices', 'max_pool2_kernel_size', 'max_pool2_stride', 'max_pool2_padding', 'max_pool2_dilation', 'max_pool2_ceil_mode', 'max_pool2_return_indices']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a 3D transposed convolution, followed by two max pooling layers and a sum operation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.max_pool1 = nn.MaxPool3d(kernel_size=2)
        self.max_pool2 = nn.MaxPool3d(kernel_size=3)

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
    # State for max_pool1 (nn.MaxPool3d)
    state_kwargs['max_pool1_kernel_size'] = model.max_pool1.kernel_size
    state_kwargs['max_pool1_stride'] = model.max_pool1.stride
    state_kwargs['max_pool1_padding'] = model.max_pool1.padding
    state_kwargs['max_pool1_dilation'] = model.max_pool1.dilation
    state_kwargs['max_pool1_ceil_mode'] = model.max_pool1.ceil_mode
    state_kwargs['max_pool1_return_indices'] = model.max_pool1.return_indices
    # State for max_pool2 (nn.MaxPool3d)
    state_kwargs['max_pool2_kernel_size'] = model.max_pool2.kernel_size
    state_kwargs['max_pool2_stride'] = model.max_pool2.stride
    state_kwargs['max_pool2_padding'] = model.max_pool2.padding
    state_kwargs['max_pool2_dilation'] = model.max_pool2.dilation
    state_kwargs['max_pool2_ceil_mode'] = model.max_pool2.ceil_mode
    state_kwargs['max_pool2_return_indices'] = model.max_pool2.return_indices
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

# ----------------------------------------------------------------------
# 1. CUDA Source: Fused transposed-conv (sum across channels) kernel
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_conv_transpose_sum_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int N, const int C_in,
    const int D_in, const int H_in, const int W_in,
    const int D_out, const int H_out, const int W_out,
    const int Kd, const int Kh, const int Kw,
    const int stride, const int padding, const int dilation)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * D_out * H_out * W_out;
    if (idx >= total_elements) return;

    int n = idx / (D_out * H_out * W_out);
    int rem = idx % (D_out * H_out * W_out);
    int d = rem / (H_out * W_out);
    int h = (rem / W_out) % H_out;
    int w = rem % W_out;

    float acc = bias[0];

    for (int c = 0; c < C_in; ++c) {
        for (int kd = 0; kd < Kd; ++kd) {
            int input_d = d + padding - kd * dilation;
            if (input_d % stride != 0) continue;
            int id = input_d / stride;
            if (id < 0 || id >= D_in) continue;

            for (int kh = 0; kh < Kh; ++kh) {
                int input_h = h + padding - kh * dilation;
                if (input_h % stride != 0) continue;
                int ih = input_h / stride;
                if (ih < 0 || ih >= H_in) continue;

                for (int kw = 0; kw < Kw; ++kw) {
                    int input_w = w + padding - kw * dilation;
                    if (input_w % stride != 0) continue;
                    int iw = input_w / stride;
                    if (iw < 0 || iw >= W_in) continue;

                    int in_ptr = ((n * C_in + c) * D_in + id) * H_in * W_in + ih * W_in + iw;
                    int wt_ptr = (c * Kd * Kh * Kw + kd * Kh * Kw + kh * Kw + kw);
                    acc += input[in_ptr] * weight[wt_ptr];
                }
            }
        }
    }
    output[idx] = acc;
}

void launch_fused_conv(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output,
    int stride, int padding, int dilation)
{
    const int N = input.size(0);
    const int C_in = input.size(1);
    const int D_in = input.size(2);
    const int H_in = input.size(3);
    const int W_in = input.size(4);
    const int D_out = output.size(2);
    const int H_out = output.size(3);
    const int W_out = output.size(4);
    const int Kd = weight.size(1);
    const int Kh = weight.size(2);
    const int Kw = weight.size(3);

    int total = N * D_out * H_out * W_out;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    fused_conv_transpose_sum_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(),
        N, C_in, D_in, H_in, W_in, D_out, H_out, W_out, Kd, Kh, Kw, stride, padding, dilation
    );
}
"""

cpp_source = r"""
void launch_fused_conv(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output, int stride, int padding, int dilation);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("launch_fused_conv", &launch_fused_conv, "Fused Transposed Conv"); }
"""

fused_ext = load_inline(name='fused_conv', cpp_sources=cpp_source, cuda_sources=cuda_source, extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True)

import torch.nn.functional as F

def functional_model(x, *, conv_transpose_weight, conv_transpose_bias, conv_transpose_stride, conv_transpose_padding, conv_transpose_output_padding, conv_transpose_groups, conv_transpose_dilation, max_pool1_kernel_size, max_pool1_stride, max_pool1_padding, max_pool1_dilation, max_pool1_ceil_mode, max_pool1_return_indices, max_pool2_kernel_size, max_pool2_stride, max_pool2_padding, max_pool2_dilation, max_pool2_ceil_mode, max_pool2_return_indices):
    # Sum weights/bias across out_channels as per logic
    w = conv_transpose_weight.sum(dim=1)
    b = conv_transpose_bias.sum(dim=0).view(1)
    N, C_in, D_in, H_in, W_in = x.shape
    Kd, Kh, Kw = w.shape[1], w.shape[2], w.shape[3]
    D_out = (D_in - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_dilation * (Kd - 1) + conv_transpose_output_padding + 1
    H_out = (H_in - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_dilation * (Kh - 1) + conv_transpose_output_padding + 1
    W_out = (W_in - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_dilation * (Kw - 1) + conv_transpose_output_padding + 1
    
    out = torch.empty((N, 1, D_out, H_out, W_out), device=x.device, dtype=x.dtype)
    fused_ext.launch_fused_conv(x, w, b, out, conv_transpose_stride, conv_transpose_padding, conv_transpose_dilation)
    
    x = F.max_pool3d(out, max_pool1_kernel_size, max_pool1_stride, max_pool1_padding, max_pool1_dilation, max_pool1_ceil_mode, max_pool1_return_indices)
    x = F.max_pool3d(x, max_pool2_kernel_size, max_pool2_stride, max_pool2_padding, max_pool2_dilation, max_pool2_ceil_mode, max_pool2_return_indices)
    return x.sum(dim=1, keepdim=True)
