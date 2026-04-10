# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_160018/code_13.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'add_value', 'multiply_value']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'add_value', 'multiply_value']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a transposed convolution, adds a value, takes the minimum, applies GELU, and multiplies by a value.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, add_value, multiply_value):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.add_value = add_value
        self.multiply_value = multiply_value

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
    if 'add_value' in flat_state:
        state_kwargs['add_value'] = flat_state['add_value']
    else:
        state_kwargs['add_value'] = getattr(model, 'add_value')
    if 'multiply_value' in flat_state:
        state_kwargs['multiply_value'] = flat_state['multiply_value']
    else:
        state_kwargs['multiply_value'] = getattr(model, 'multiply_value')
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

# Optimization: Implemented custom CUDA kernel for 2D Transposed Convolution
# and fused the subsequent element-wise operations (add, min, gelu, multiply).
# Note: For simplicity and performance in a single-file demonstration, 
# the conv_transpose2d logic is handled by a fused kernel approach.

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

// Simple implementation of Transposed Conv2D + Fused activation Kernel
// In a high-performance setting, one would use cuDNN, but as per requirements, 
// we implement the logic directly in CUDA.
__global__ void fused_conv_transpose_activation_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch, int in_c, int in_h, int in_w,
    int out_c, int out_h, int out_w,
    int k_h, int k_w, int stride, int padding,
    float add_v, float mul_v
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= batch * out_c * out_h * out_w) return;

    int tmp = out_idx;
    int ow = tmp % out_w; tmp /= out_w;
    int oh = tmp % out_h; tmp /= out_h;
    int oc = tmp % out_c; tmp /= out_c;
    int b = tmp;

    float val = bias[oc];

    for (int ic = 0; ic < in_c; ++ic) {
        for (int kh = 0; kh < k_h; ++kh) {
            for (int kw = 0; kw < k_w; ++kw) {
                int ih = (oh + padding - kh);
                int iw = (ow + padding - kw);
                if (ih % stride == 0 && iw % stride == 0) {
                    ih /= stride; iw /= stride;
                    if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                        float w = weight[ic * out_c * k_h * k_w + oc * k_h * k_w + kh * k_w + kw];
                        val += input[(((b * in_c + ic) * in_h + ih) * in_w + iw)] * w;
                    }
                }
            }
        }
    }

    // Fused ops
    val += add_v;
    val = fminf(val, 0.0f);
    float cdf = 0.5f * (1.0f + tanhf(0.7978845608028654f * (val + 0.044715f * val * val * val)));
    val = (val * cdf) * mul_v;
    
    output[out_idx] = val;
}

void launch_fused_conv(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output,
    int stride, int padding, float add_v, float mul_v
) {
    int b = input.size(0), in_c = input.size(1), in_h = input.size(2), in_w = input.size(3);
    int out_c = weight.size(1);
    int k_h = weight.size(2), k_w = weight.size(3);
    int out_h = (in_h - 1) * stride - 2 * padding + k_h;
    int out_w = (in_w - 1) * stride - 2 * padding + k_w;
    
    int numel = b * out_c * out_h * out_w;
    int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    
    fused_conv_transpose_activation_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), b, in_c, in_h, in_w, out_c, out_h, out_w,
        k_h, k_w, stride, padding, add_v, mul_v
    );
}
"""

cpp_source = r"""
void launch_fused_conv(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int, int, float, float);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_forward", &launch_fused_conv, "Fused ConvTranspose + Activation");
}
"""

fused_ext = load_inline(name='fused_op', cpp_sources=cpp_source, cuda_sources=cuda_kernel, extra_cuda_cflags=['-O3'], with_cuda=True)

def functional_model(x, *, conv_transpose_weight, conv_transpose_bias, conv_transpose_stride, 
                     conv_transpose_padding, conv_transpose_output_padding, conv_transpose_groups, 
                     conv_transpose_dilation, add_value, multiply_value):
    out_h = (x.size(2) - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_weight.size(2) + conv_transpose_output_padding
    out_w = (x.size(3) - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_weight.size(3) + conv_transpose_output_padding
    output = torch.empty((x.size(0), conv_transpose_weight.size(1), out_h, out_w), device=x.device)
    fused_ext.fused_forward(x, conv_transpose_weight, conv_transpose_bias, output, 
                           conv_transpose_stride, conv_transpose_padding, float(add_value), float(multiply_value))
    return output
