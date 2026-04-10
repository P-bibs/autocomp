# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_155151/code_12.py
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

# Optimized CUDA kernel implementing Transposed Convolution + Fused Post-ops
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__device__ __forceinline__ float fast_gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

__global__ void fused_conv_transpose_gelu_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size, int in_channels, int out_channels,
    int in_h, int in_w, int out_h, int out_w,
    int kernel_h, int kernel_w, int stride, int padding,
    float add_val, float mul_val
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= batch_size * out_channels * out_h * out_w) return;

    // Index mapping for NHWC or NCHW layout (Assuming NCHW)
    int tmp = out_idx;
    int ow = tmp % out_w; tmp /= out_w;
    int oh = tmp % out_h; tmp /= out_h;
    int oc = tmp % out_channels; tmp /= out_channels;
    int b = tmp;

    float val = (bias != nullptr) ? bias[oc] : 0.0f;

    // Optimized Transposed Conv loop
    // Output at (oh, ow) is sum of inputs * weights
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int ih = (oh + padding - kh);
                int iw = (ow + padding - kw);
                
                if (ih % stride == 0 && iw % stride == 0) {
                    ih /= stride;
                    iw /= stride;
                    if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                        float in_val = input[((b * in_channels + ic) * in_h + ih) * in_w + iw];
                        float w_val = weight[((ic * out_channels + oc) * kernel_h + kh) * kernel_w + kw];
                        val += in_val * w_val;
                    }
                }
            }
        }
    }

    // Fused non-linear ops
    val = fast_gelu(fminf(val + add_val, 0.0f)) * mul_val;
    output[out_idx] = val;
}

void fused_conv_transpose_gelu_forward(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output,
    int stride, int padding, float add_val, float mul_val
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_h = input.size(2);
    int in_w = input.size(3);
    int out_channels = weight.size(1);
    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);
    int out_h = output.size(2);
    int out_w = output.size(3);

    int total_elements = batch_size * out_channels * out_h * out_w;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    fused_conv_transpose_gelu_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(),
        bias.numel() > 0 ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels, in_h, in_w, out_h, out_w,
        kernel_h, kernel_w, stride, padding, add_val, mul_val
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_conv_transpose_gelu_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output, int stride, int padding, float add_val, float mul_val);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused", &fused_conv_transpose_gelu_forward, "Fused custom conv_transpose2d");
}
"""

fused_ext = load_inline(name='fused_ext', cpp_sources=cpp_source, cuda_sources=cuda_kernel, with_cuda=True, extra_cuda_cflags=['-O3'])

def functional_model(x, *, conv_transpose_weight, conv_transpose_bias, conv_transpose_stride, 
                     conv_transpose_padding, conv_transpose_output_padding, conv_transpose_groups, 
                     conv_transpose_dilation, add_value, multiply_value):
    batch_size, in_channels, in_h, in_w = x.shape
    out_channels = conv_transpose_weight.shape[1]
    stride = conv_transpose_stride
    pad = conv_transpose_padding
    out_h = (in_h - 1) * stride - 2 * pad + conv_transpose_weight.shape[2] + conv_transpose_output_padding
    out_w = (in_w - 1) * stride - 2 * pad + conv_transpose_weight.shape[3] + conv_transpose_output_padding
    
    out = torch.empty(batch_size, out_channels, out_h, out_w, device='cuda', dtype=torch.float32)
    fused_ext.fused(x, conv_transpose_weight, conv_transpose_bias, out, stride, pad, float(add_value), float(multiply_value))
    return out
