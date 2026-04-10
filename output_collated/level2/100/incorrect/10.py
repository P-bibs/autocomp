# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_115141/code_4.py
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

# CUDA kernel for fused conv_transpose3d + clamp + div
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void fused_conv_transpose3d_clamp_div_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_d, int input_h, int input_w,
    int output_d, int output_h, int output_w,
    int kernel_size, int stride, int padding,
    float min_value, float divisor) {
    
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * output_d * output_h * output_w;
    
    if (out_idx >= total_elements) return;
    
    int temp = out_idx;
    int w = temp % output_w; temp /= output_w;
    int h = temp % output_h; temp /= output_h;
    int d = temp % output_d; temp /= output_d;
    int oc = temp % out_channels;
    int b = temp / out_channels;
    
    float val = (bias != nullptr) ? bias[oc] : 0.0f;
    
    // Transpose conv logic: each input pixel (ic, id, ih, iw) contributes to a kernel-sized 3D box in output
    // Here we compute the value by gathering inputs that contribute to the current output pixel
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kd = 0; kd < kernel_size; ++kd) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    
                    int id = (d + padding - kd);
                    int ih = (h + padding - kh);
                    int iw = (w + padding - kw);
                    
                    if (id % stride == 0 && ih % stride == 0 && iw % stride == 0) {
                        id /= stride; ih /= stride; iw /= stride;
                        
                        if (id >= 0 && id < input_d && ih >= 0 && ih < input_h && iw >= 0 && iw < input_w) {
                            int in_pos = ((b * in_channels + ic) * input_d + id) * (input_h * input_w) + ih * input_w + iw;
                            int w_pos = ((oc * in_channels + ic) * kernel_size + kd) * (kernel_size * kernel_size) + kh * kernel_size + kw;
                            val += input[in_pos] * weight[w_pos];
                        }
                    }
                }
            }
        }
    }
    
    output[out_idx] = fmaxf(min_value, val) / divisor;
}

void fused_conv_transpose3d_clamp_div(
    const torch::Tensor input,
    const torch::Tensor weight,
    const torch::Tensor bias,
    torch::Tensor output,
    int kernel_size, int stride, int padding,
    float min_value, float divisor) {
    
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int out_channels = weight.size(1);
    int input_d = input.size(2);
    int input_h = input.size(3);
    int input_w = input.size(4);
    int output_d = output.size(2);
    int output_h = output.size(3);
    int output_w = output.size(4);
    
    int total_elements = batch_size * out_channels * output_d * output_h * output_w;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    
    fused_conv_transpose3d_clamp_div_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(),
        bias.numel() > 0 ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        input_d, input_h, input_w,
        output_d, output_h, output_w,
        kernel_size, stride, padding,
        min_value, divisor
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_conv_transpose3d_clamp_div(const torch::Tensor, const torch::Tensor, const torch::Tensor, torch::Tensor, int, int, int, float, float);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_forward", &fused_conv_transpose3d_clamp_div, "Fused ConvTranspose3d kernel");
}
"""

module = load_inline(name='fused_ct3d', cpp_sources=cpp_source, cuda_sources=cuda_kernel, extra_cuda_cflags=['-O3'])

def functional_model(x, *, conv_transpose_weight, conv_transpose_bias, conv_transpose_stride, conv_transpose_padding, conv_transpose_output_padding, conv_transpose_groups, conv_transpose_dilation, min_value, divisor):
    b, ic, id, ih, iw = x.shape
    oc, _, kd, kh, kw = conv_transpose_weight.shape
    stride, pad = conv_transpose_stride[0], conv_transpose_padding[0]
    out_d = (id - 1) * stride - 2 * pad + kd + conv_transpose_output_padding[0]
    out_h = (ih - 1) * stride - 2 * pad + kh + conv_transpose_output_padding[1]
    out_w = (iw - 1) * stride - 2 * pad + kw + conv_transpose_output_padding[2]
    
    output = torch.empty((b, oc, out_d, out_h, out_w), device=x.device, dtype=x.dtype)
    module.fused_forward(x.contiguous(), conv_transpose_weight.contiguous(), 
                         conv_transpose_bias if conv_transpose_bias is not None else torch.tensor([], device=x.device),
                         output, kd, stride, pad, min_value, divisor)
    return output
