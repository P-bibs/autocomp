# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_123254/code_18.py
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

# CUDA Kernel: Fuses ConvTranspose2d, bias addition, bias subtraction, and Tanh
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_conv_transpose_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ act_bias,
    float* __restrict__ output,
    int N, int in_C, int out_C,
    int in_H, int in_W,
    int out_H, int out_W,
    int k_size, int stride, int padding
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * out_C * out_H * out_W;
    if (idx >= total) return;

    int n = idx / (out_C * out_H * out_W);
    int c = (idx / (out_H * out_W)) % out_C;
    int oh = (idx / out_W) % out_H;
    int ow = idx % out_W;

    float acc = (conv_bias != nullptr) ? conv_bias[c] : 0.0f;

    // Direct convolution transpose implementation
    for (int ic = 0; ic < in_C; ++ic) {
        for (int kh = 0; kh < k_size; ++kh) {
            for (int kw = 0; kw < k_size; ++kw) {
                int ih = (oh + padding - kh);
                int iw = (ow + padding - kw);
                if (ih >= 0 && ih < in_H * stride && iw >= 0 && iw < in_W * stride && 
                    ih % stride == 0 && iw % stride == 0) {
                    int i_idx = ((n * in_C + ic) * in_H + (ih / stride)) * in_W + (iw / stride);
                    int w_idx = ((c * in_C + ic) * k_size + kh) * k_size + kw;
                    acc += input[i_idx] * weight[w_idx];
                }
            }
        }
    }
    output[idx] = tanhf(acc - act_bias[c]);
}

void fused_forward(
    const torch::Tensor& x, const torch::Tensor& weight, 
    const torch::Tensor& c_bias, const torch::Tensor& a_bias, 
    torch::Tensor& output, int stride, int padding
) {
    int N = x.size(0);
    int in_C = x.size(1);
    int in_H = x.size(2);
    int in_W = x.size(3);
    int out_C = weight.size(1);
    int k_size = weight.size(2);
    int out_H = (in_H - 1) * stride - 2 * padding + k_size;
    int out_W = (in_W - 1) * stride - 2 * padding + k_size;
    
    int total = N * out_C * out_H * out_W;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    
    fused_conv_transpose_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(), weight.data_ptr<float>(), 
        c_bias.data_ptr<float>(), a_bias.data_ptr<float>(),
        output.data_ptr<float>(), N, in_C, out_C, in_H, in_W, out_H, out_W, k_size, stride, padding
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_forward(const torch::Tensor& x, const torch::Tensor& weight, const torch::Tensor& c_bias, const torch::Tensor& a_bias, torch::Tensor& output, int stride, int padding);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_forward", &fused_forward, "Fused ConvTranspose + Bias + Tanh");
}
"""

fused_ext = load_inline(name='fused_ext', cpp_sources=cpp_source, cuda_sources=cuda_source, extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True)

def functional_model(x, *, conv_transpose_weight, conv_transpose_bias, conv_transpose_stride, conv_transpose_padding, conv_transpose_output_padding, conv_transpose_groups, conv_transpose_dilation, bias):
    # Only group=1, dilation=1 supported for simplified custom kernel performance
    out_C = conv_transpose_weight.size(1)
    out_H = (x.size(2) - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_weight.size(2) + conv_transpose_output_padding
    out_W = (x.size(3) - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_weight.size(3) + conv_transpose_output_padding
    
    output = torch.empty((x.size(0), out_C, out_H, out_W), device=x.device, dtype=x.dtype)
    fused_ext.fused_forward(x, conv_transpose_weight, conv_transpose_bias, bias.view(-1), output, conv_transpose_stride, conv_transpose_padding)
    return output
