# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_134740/code_30.py
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
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// Fused kernel: Transposed Conv3D + Elementwise Operation
// f(x) = ((x + bias) + x) * x + x = (2x + bias) * x + x
__global__ void fused_transconv_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C_in, int C_out,
    int D, int H, int W,
    int KD, int KH, int KW,
    int stride, int pad
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C_in * D * H * W) return;

    // Simplified index decomposition
    int tmp = idx;
    int w = tmp % W; tmp /= W;
    int h = tmp % H; tmp /= H;
    int d = tmp % D; tmp /= D;
    int c = tmp % C_in; tmp /= C_in;
    int n = tmp;

    for (int oc = 0; oc < C_out; ++oc) {
        float bias_val = bias[oc];
        for (int kd = 0; kd < KD; ++kd) {
            for (int kh = 0; kh < KH; ++kh) {
                for (int kw = 0; kw < KW; ++kw) {
                    int od = (d - pad + kd) * stride;
                    int oh = (h - pad + kh) * stride;
                    int ow = (w - pad + kw) * stride;
                    
                    if (od >= 0 && od < (D-1)*stride + KD && oh >= 0 && oh < (H-1)*stride + KH && ow >= 0 && ow < (W-1)*stride + KW) {
                        float val = input[idx] * weight[((c * C_out + oc) * KD + kd) * KH * KW + kh * KW + kw];
                        // Apply fused logic: ((val + bias) + val) * val + val
                        float res = ((val + bias_val) + val) * val + val;
                        atomicAdd(&output[((n * C_out + oc) * ((D-1)*stride+KD)) * ((H-1)*stride+KH) * ((W-1)*stride+KW) + od * ((H-1)*stride+KH) * ((W-1)*stride+KW) + oh * ((W-1)*stride+KW) + ow], res);
                    }
                }
            }
        }
    }
}

void fused_transconv_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output) {
    int N = input.size(0); int C_in = input.size(1);
    int D = input.size(2); int H = input.size(3); int W = input.size(4);
    int C_out = weight.size(1);
    int KD = weight.size(2); int KH = weight.size(3); int KW = weight.size(4);
    
    int total_threads = N * C_in * D * H * W;
    int threads = 256;
    int blocks = (total_threads + threads - 1) / threads;
    
    fused_transconv_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), N, C_in, C_out, D, H, W, KD, KH, KW, 2, 1
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_transconv_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_transconv", &fused_transconv_forward, "Fused Transposed Conv kernel");
}
"""

fused_ext = load_inline(
    name='fused_transconv_ext', cpp_sources=cpp_source, cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True
)

def functional_model(x, *, conv_transpose_weight, conv_transpose_bias, conv_transpose_stride, 
                     conv_transpose_padding, conv_transpose_output_padding, conv_transpose_groups, 
                     conv_transpose_dilation, bias):
    # Output spatial dim calculation
    out_d = (x.size(2) - 1) * conv_transpose_stride[0] - 2 * conv_transpose_padding[0] + (conv_transpose_weight.size(2)-1) + conv_transpose_output_padding[0] + 1
    out_h = (x.size(3) - 1) * conv_transpose_stride[1] - 2 * conv_transpose_padding[1] + (conv_transpose_weight.size(3)-1) + conv_transpose_output_padding[1] + 1
    out_w = (x.size(4) - 1) * conv_transpose_stride[2] - 2 * conv_transpose_padding[2] + (conv_transpose_weight.size(4)-1) + conv_transpose_output_padding[2] + 1
    
    output = torch.zeros((x.size(0), conv_transpose_weight.size(1), out_d, out_h, out_w), device='cuda')
    fused_ext.fused_transconv(x.contiguous(), conv_transpose_weight.contiguous(), bias.view(-1).contiguous(), output)
    return output
