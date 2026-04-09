# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_082041/code_13.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_weight', 'conv_bias', 'conv_stride', 'conv_padding', 'conv_dilation', 'conv_groups']
REQUIRED_FLAT_STATE_NAMES = ['conv_weight', 'conv_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a convolution, applies minimum operation, Tanh, and another Tanh.
    """

    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

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
    # State for conv (nn.Conv2d)
    if 'conv_weight' in flat_state:
        state_kwargs['conv_weight'] = flat_state['conv_weight']
    else:
        state_kwargs['conv_weight'] = getattr(model.conv, 'weight', None)
    if 'conv_bias' in flat_state:
        state_kwargs['conv_bias'] = flat_state['conv_bias']
    else:
        state_kwargs['conv_bias'] = getattr(model.conv, 'bias', None)
    state_kwargs['conv_stride'] = model.conv.stride
    state_kwargs['conv_padding'] = model.conv.padding
    state_kwargs['conv_dilation'] = model.conv.dilation
    state_kwargs['conv_groups'] = model.conv.groups
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

# The custom CUDA kernel performs the convolution loop manually
# to avoid using cuDNN/LibTorch built-in convolution routines.
# It fuses the convolution output reduction (min) and activation (tanh).
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>

__global__ void fused_conv2d_min_tanh_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_height,
    int input_width,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    int groups,
    int output_height,
    int output_width
) {
    int batch_idx = blockIdx.z;
    int out_h = blockIdx.y * blockDim.y + threadIdx.y;
    int out_w = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx >= batch_size || out_h >= output_height || out_w >= output_width) return;
    
    int channels_per_group = in_channels / groups;
    int out_channels_per_group = out_channels / groups;
    
    float min_val = FLT_MAX;
    
    for (int oc = 0; oc < out_channels; oc++) {
        int group_id = oc / out_channels_per_group;
        float conv_result = bias[oc];
        
        for (int ic = 0; ic < channels_per_group; ic++) {
            int in_channel = group_id * channels_per_group + ic;
            for (int kh = 0; kh < kernel_size; kh++) {
                for (int kw = 0; kw < kernel_size; kw++) {
                    int in_h = out_h * stride + kh * dilation - padding;
                    int in_w = out_w * stride + kw * dilation - padding;
                    
                    if (in_h >= 0 && in_h < input_height && in_w >= 0 && in_w < input_width) {
                        int i_idx = (((batch_idx * in_channels + in_channel) * input_height + in_h) * input_width + in_w);
                        int w_idx = (((oc * channels_per_group + ic) * kernel_size + kh) * kernel_size + kw);
                        conv_result += input[i_idx] * weight[w_idx];
                    }
                }
            }
        }
        if (conv_result < min_val) min_val = conv_result;
    }
    
    int o_idx = (((batch_idx * 1 + 0) * output_height + out_h) * output_width + out_w);
    output[o_idx] = tanhf(min_val);
}

void fused_op_forward(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output,
    int stride, int padding, int dilation, int groups
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int out_channels = weight.size(0);
    int input_height = input.size(2);
    int input_width = input.size(3);
    int kernel_size = weight.size(2);
    int output_height = output.size(2);
    int output_width = output.size(3);
    
    dim3 threads(16, 16);
    dim3 blocks((output_width + 15) / 16, (output_height + 15) / 16, batch_size);
    
    fused_conv2d_min_tanh_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(),
        batch_size, in_channels, out_channels, input_height, input_width,
        kernel_size, stride, padding, dilation, groups,
        output_height, output_width
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_op_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output, int stride, int padding, int dilation, int groups);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward);
}
"""

module = load_inline(name='fused_module', cpp_sources=cpp_source, cuda_sources=cuda_kernel, with_cuda=True, extra_cuda_cflags=['-O3'])

def functional_model(x, *, conv_weight, conv_bias, conv_stride, conv_padding, conv_dilation, conv_groups):
    oh = (x.size(2) + 2 * conv_padding - conv_dilation * (conv_weight.size(2) - 1) - 1) // conv_stride + 1
    ow = (x.size(3) + 2 * conv_padding - conv_dilation * (conv_weight.size(3) - 1) - 1) // conv_stride + 1
    output = torch.empty((x.size(0), 1, oh, ow), device=x.device, dtype=x.dtype)
    module.fused_op(x.contiguous(), conv_weight.contiguous(), conv_bias, output, conv_stride, conv_padding, conv_dilation, conv_groups)
    return output
