# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_123254/code_30.py
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

# CUDA Kernel: Fused Transposed Convolution + Bias Subtraction + Tanh
# We use a 2D grid mapping over output channels and spatial locations.
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_conv_transpose_tanh_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int in_C, int in_H, int in_W,
    int out_C, int out_H, int out_W,
    int kH, int kW, int stride, int padding) 
{
    int out_c = blockIdx.z;
    int out_i = blockIdx.y * blockDim.y + threadIdx.y;
    int out_j = blockIdx.x * blockDim.x + threadIdx.x;

    if (out_i < out_H && out_j < out_W) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int in_c = 0; in_c < in_C; ++in_c) {
                for (int kh = 0; kh < kH; ++kh) {
                    for (int kw = 0; kw < kW; ++kw) {
                        int in_i = out_i + padding - kh;
                        int in_j = out_j + padding - kw;
                        if (in_i % stride == 0 && in_j % stride == 0) {
                            int r_i = in_i / stride;
                            int r_j = in_j / stride;
                            if (r_i >= 0 && r_i < in_H && r_j >= 0 && r_j < in_W) {
                                int input_idx = ((n * in_C + in_c) * in_H + r_i) * in_W + r_j;
                                int weight_idx = ((out_c * in_C + in_c) * kH + kh) * kW + kw;
                                sum += input[input_idx] * weight[weight_idx];
                            }
                        }
                    }
                }
            }
            int out_idx = ((n * out_C + out_c) * out_H + out_i) * out_W + out_j;
            output[out_idx] = tanhf(sum - bias[out_c]);
        }
    }
}

void fused_op_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, torch::Tensor output, 
                      int stride, int padding) {
    int N = x.size(0); int in_C = x.size(1); int in_H = x.size(2); int in_W = x.size(3);
    int out_C = bias.size(0); int out_H = output.size(2); int out_W = output.size(3);
    int kH = weight.size(2); int kW = weight.size(3);

    dim3 threads(16, 16);
    dim3 blocks((out_W + 15) / 16, (out_H + 15) / 16, out_C);

    fused_conv_transpose_tanh_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), 
        output.data_ptr<float>(), N, in_C, in_H, in_W, out_C, out_H, out_W, kH, kW, stride, padding
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_op_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, torch::Tensor output, int stride, int padding);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused Transposed Conv + Tanh");
}
"""

fused_ext = load_inline(name='fused_ext', cpp_sources=cpp_source, cuda_sources=cuda_source,
                        extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True)

def functional_model(x, *, conv_transpose_weight, conv_transpose_bias, conv_transpose_stride,
                     conv_transpose_padding, conv_transpose_output_padding, conv_transpose_groups,
                     conv_transpose_dilation, bias):
    x, w = x.cuda(), conv_transpose_weight.cuda()
    b = bias.view(-1).cuda()
    
    # Calculate output dimensions
    out_h = (x.size(2) - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_weight.size(2) + conv_transpose_output_padding[0]
    out_w = (x.size(3) - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_weight.size(3) + conv_transpose_output_padding[1]
    output = torch.empty((x.size(0), b.size(0), out_h, out_w), device='cuda')
    
    fused_ext.fused_op(x, w, b, output, conv_transpose_stride, conv_transpose_padding)
    return output
