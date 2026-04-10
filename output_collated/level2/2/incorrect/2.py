# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_161645/code_4.py
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

# CUDA kernel for fused transposed convolution, bias, and scaling
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_transpose_conv_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int B, int Ci, int Co, int Hi, int Wi, int Ho, int Wo,
    int k, int s, int p, float scale
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= B * Co * Ho * Wo) return;

    int tmp = tid;
    int w_out = tmp % Wo; tmp /= Wo;
    int h_out = tmp % Ho; tmp /= Ho;
    int c_out = tmp % Co; tmp /= Co;
    int b = tmp;

    float acc = bias[c_out];

    // Transposed Convolution logic:
    // For each output pixel, we check if it falls under the receptive field of each input pixel
    for (int c_in = 0; c_in < Ci; ++c_in) {
        for (int kh = 0; kh < k; ++kh) {
            for (int kw = 0; kw < k; ++kw) {
                int h_in = h_out + p - kh;
                int w_in = w_out + p - kw;

                if (h_in % s == 0 && w_in % s == 0) {
                    h_in /= s;
                    w_in /= s;
                    if (h_in >= 0 && h_in < Hi && w_in >= 0 && w_in < Wi) {
                        float val = input[(((b * Ci + c_in) * Hi + h_in) * Wi + w_in)];
                        float w = weight[((c_out * Ci + c_in) * k + kh) * k + kw];
                        acc += val * w;
                    }
                }
            }
        }
    }

    // Fused element-wise operations
    acc = fmaxf(0.0f, fminf(1.0f, acc));
    acc *= scale;
    acc = fmaxf(0.0f, fminf(1.0f, acc));
    acc /= scale;

    output[tid] = acc;
}

void fused_op_forward(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output,
    int B, int Ci, int Co, int Hi, int Wi, int Ho, int Wo, int k, int s, int p, float scale
) {
    int total = B * Co * Ho * Wo;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    fused_transpose_conv_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), B, Ci, Co, Hi, Wi, Ho, Wo, k, s, p, scale
    );
}
"""

cpp_source = r"""
void fused_op_forward(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output,
    int B, int Ci, int Co, int Hi, int Wi, int Ho, int Wo, int k, int s, int p, float scale
);
"""

fused_ext = load_inline(
    name='fused_transpose_conv',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    functions=['fused_op_forward']
)

def functional_model(x, *, conv_transpose_weight, conv_transpose_bias, conv_transpose_stride, 
                     conv_transpose_padding, conv_transpose_output_padding, conv_transpose_groups, 
                     conv_transpose_dilation, bias, scaling_factor):
    B, Ci, Hi, Wi = x.shape
    Co, _, k, _ = conv_transpose_weight.shape
    s, p, op = conv_transpose_stride, conv_transpose_padding, conv_transpose_output_padding
    
    Ho = (Hi - 1) * s - 2 * p + k + op
    Wo = (Wi - 1) * s - 2 * p + k + op
    
    output = torch.empty((B, Co, Ho, Wo), device=x.device, dtype=x.dtype)
    
    fused_ext.fused_op_forward(
        x, conv_transpose_weight, bias.squeeze(), output,
        B, Ci, Co, Hi, Wi, Ho, Wo, k, s, p, float(scaling_factor)
    )
    return output
