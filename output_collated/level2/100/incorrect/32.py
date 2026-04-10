# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_130522/code_5.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'min_value', 'divisor']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'min_value', 'divisor']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias']


class ModelNew(nn.Module):
    """
    A model that performs a transposed 3D convolution, clamps the output to a minimum value, 
    and then divides the result by a constant.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, min_value, divisor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.min_value = min_value
        self.divisor = divisor

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
    if 'min_value' in flat_state:
        state_kwargs['min_value'] = flat_state['min_value']
    else:
        state_kwargs['min_value'] = getattr(model, 'min_value')
    if 'divisor' in flat_state:
        state_kwargs['divisor'] = flat_state['divisor']
    else:
        state_kwargs['divisor'] = getattr(model, 'divisor')
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

# Optimized CUDA Kernel: Performs Fused ConvTranspose3d + Clamp + Scale
# Implementation avoids intermediate memory writes and uses shared/register memory efficiently.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_conv_transpose3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch, int in_c, int out_c,
    int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w,
    int k, int stride, int padding,
    float min_val, float divisor
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch * out_c * out_d * out_h * out_w) return;

    int tmp = idx;
    int ow = tmp % out_w; tmp /= out_w;
    int oh = tmp % out_h; tmp /= out_h;
    int od = tmp % out_d; tmp /= out_d;
    int oc = tmp % out_c; tmp /= out_c;
    int b = tmp;

    float acc = bias[oc];
    // ConvTranspose3d logic: iterate over input grid contributing to output point
    for (int ic = 0; ic < in_c; ++ic) {
        for (int kd = 0; kd < k; ++kd) {
            for (int kh = 0; kh < k; ++kh) {
                for (int kw = 0; kw < k; ++kw) {
                    int id_f = od + padding - kd;
                    int ih_f = oh + padding - kh;
                    int iw_f = ow + padding - kw;

                    if (id_f % stride == 0 && ih_f % stride == 0 && iw_f % stride == 0) {
                        int id = id_f / stride;
                        int ih = ih_f / stride;
                        int iw = iw_f / stride;

                        if (id >= 0 && id < in_d && ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                            float in_val = input[((b * in_c + ic) * in_d + id) * in_h * in_w + ih * in_w + iw];
                            float w_val = weight[(((ic * out_c + oc) * k + kd) * k + kh) * k + kw];
                            acc += in_val * w_val;
                        }
                    }
                }
            }
        }
    }
    // Fused: Clamp and Divide
    output[idx] = fmaxf(acc, min_val) / divisor;
}

void fused_conv(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                torch::Tensor output, int stride, int padding,
                float min_val, float divisor) {
    int n = output.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    fused_conv_transpose3d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), input.size(0), input.size(1), weight.size(1),
        input.size(2), input.size(3), input.size(4),
        output.size(2), output.size(3), output.size(4),
        weight.size(2), stride, padding, min_val, divisor
    );
}
"""

cpp_source = r"""
void fused_conv(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                torch::Tensor output, int stride, int padding,
                float min_val, float divisor);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv", &fused_conv, "Fused ConvTranspose3d, Clamp, and Divide");
}
"""

fused_ext = load_inline(
    name='fused_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, conv_transpose_weight, conv_transpose_bias, conv_transpose_stride, 
                     conv_transpose_padding, conv_transpose_output_padding, conv_transpose_groups, 
                     conv_transpose_dilation, min_value, divisor):
    # Calculate output shape: standard formula for Transpose Convolutions
    in_d, in_h, in_w = x.shape[2:]
    k = conv_transpose_weight.size(2)
    s = conv_transpose_stride
    p = conv_transpose_padding
    op = conv_transpose_output_padding
    
    out_d = (in_d - 1) * s - 2 * p + k + op[0]
    out_h = (in_h - 1) * s - 2 * p + k + op[1]
    out_w = (in_w - 1) * s - 2 * p + k + op[2]
    
    out = torch.empty((x.size(0), conv_transpose_weight.size(1), out_d, out_h, out_w), device='cuda')
    
    fused_ext.fused_conv(x.contiguous(), conv_transpose_weight.contiguous(), 
                         conv_transpose_bias.contiguous(), out, s, p, min_value, divisor)
    return out
