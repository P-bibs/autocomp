# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_133559/code_11.py
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
#include <math.h>

__global__ void fused_conv_transpose_postproc_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ post_bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int in_d,
    const int in_h,
    const int in_w,
    const int out_d,
    const int out_h,
    const int out_w,
    const int kernel_d,
    const int kernel_h,
    const int kernel_w,
    const int stride_d,
    const int stride_h,
    const int stride_w,
    const int pad_d,
    const int pad_h,
    const int pad_w,
    const int out_pad_d,
    const int out_pad_h,
    const int out_pad_w,
    const int groups
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int spatial_size = out_d * out_h * out_w;
    int total_out_elements = batch_size * out_channels * spatial_size;
    
    if (out_idx >= total_out_elements) return;
    
    // Decompose linear index to (batch, channel, d, h, w)
    int remaining = out_idx;
    int b = remaining / (out_channels * spatial_size);
    remaining %= (out_channels * spatial_size);
    int c = remaining / spatial_size;
    remaining %= spatial_size;
    int od = remaining / (out_h * out_w);
    remaining %= (out_h * out_w);
    int oh = remaining / out_w;
    int ow = remaining % out_w;
    
    // Accumulate convolution result
    float acc = 0.0f;
    int group_size = in_channels / groups;
    int group_id = c / (out_channels / groups);
    int c_offset = group_id * group_size;
    
    // Iterate over input spatial dimensions
    for (int ic = 0; ic < group_size; ic++) {
        for (int kd = 0; kd < kernel_d; kd++) {
            for (int kh = 0; kh < kernel_h; kh++) {
                for (int kw = 0; kw < kernel_w; kw++) {
                    // Compute corresponding input position
                    int id = od - out_pad_d + kd * stride_d - pad_d;
                    int ih = oh - out_pad_h + kh * stride_h - pad_h;
                    int iw = ow - out_pad_w + kw * stride_w - pad_w;
                    
                    // Check bounds
                    if (id >= 0 && id < in_d && ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                        int in_idx = ((b * in_channels + c_offset + ic) * in_d + id) * in_h * in_w + ih * in_w + iw;
                        int w_idx = ((c * group_size + ic) * kernel_d + kd) * kernel_h * kernel_w + kh * kernel_w + kw;
                        acc += input[in_idx] * weight[w_idx];
                    }
                }
            }
        }
    }
    
    // Add convolution bias
    acc += conv_bias[c];
    
    // Apply post-processing: out = x * (2*x + post_bias + 1)
    float x = acc;
    float bias_val = post_bias[c];
    float result = x * (2.0f * x + bias_val + 1.0f);
    
    output[out_idx] = result;
}

void fused_conv_transpose_postproc_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& conv_bias,
    const torch::Tensor& post_bias,
    torch::Tensor& output,
    const int stride_d,
    const int stride_h,
    const int stride_w,
    const int pad_d,
    const int pad_h,
    const int pad_w,
    const int out_pad_d,
    const int out_pad_h,
    const int out_pad_w,
    const int groups
) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto in_d = input.size(2);
    auto in_h = input.size(3);
    auto in_w = input.size(4);
    
    auto out_channels = weight.size(1) * groups;
    auto kernel_d = weight.size(2);
    auto kernel_h = weight.size(3);
    auto kernel_w = weight.size(4);
    
    auto out_d = output.size(2);
    auto out_h = output.size(3);
    auto out_w = output.size(4);
    
    int total_elements = batch_size * out_channels * out_d * out_h * out_w;
    
    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;
    
    fused_conv_transpose_postproc_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        conv_bias.data_ptr<float>(),
        post_bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        in_d, in_h, in_w,
        out_d, out_h, out_w,
        kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        out_pad_d, out_pad_h, out_pad_w,
        groups
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_conv_transpose_postproc_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& conv_bias,
    const torch::Tensor& post_bias,
    torch::Tensor& output,
    const int stride_d,
    const int stride_h,
    const int stride_w,
    const int pad_d,
    const int pad_h,
    const int pad_w,
    const int out_pad_d,
    const int out_pad_h,
    const int out_pad_w,
    const int groups
);

torch::Tensor fused_conv_transpose_postproc(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& conv_bias,
    const torch::Tensor& post_bias,
    const int stride_d,
    const int stride_h,
    const int stride_w,
    const int pad_d,
    const int pad_h,
    const int pad_w,
    const int out_pad_d,
    const int out_pad_h,
    const int out_pad_w,
    const int groups
) {
    auto batch_size = input.size(0);
    auto out_channels = weight.size(1) * groups;
    
    // Compute output spatial dimensions
    auto in_d = input.size(2);
    auto in_h = input.size(3);
    auto in_w = input.size(4);
    auto kernel_d = weight.size(2);
    auto kernel_h = weight.size(3);
    auto kernel_w = weight.size(4);
    
    int out_d = (in_d - 1) * stride_d - 2 * pad_d + kernel_d + out_pad_d;
    int out_h = (in_h - 1) * stride_h - 2 * pad_h + kernel_h + out_pad_h;
    int out_w = (in_w - 1) * stride_w - 2 * pad_w + kernel_w + out_pad_w;
    
    auto output = torch::empty({batch_size, out_channels, out_d, out_h, out_w}, input.options());
    
    fused_conv_transpose_postproc_forward(
        input, weight, conv_bias, post_bias, output,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        out_pad_d, out_pad_h, out_pad_w,
        groups
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_transpose_postproc", &fused_conv_transpose_postproc,
          "Fused transpose convolution with post-processing in a single kernel");
}
"""

fused_ext = load_inline(
    name='fused_conv_transpose_postproc_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(
    x,
    *,
    conv_transpose_weight,
    conv_transpose_bias,
    conv_transpose_stride,
    conv_transpose_padding,
    conv_transpose_output_padding,
    conv_transpose_groups,
    conv_transpose_dilation,
    bias,
):
    # Use fused kernel for both transpose convolution and post-processing
    stride_d, stride_h, stride_w = conv_transpose_stride
    pad_d, pad_h, pad_w = conv_transpose_padding
    out_pad_d, out_pad_h, out_pad_w = conv_transpose_output_padding
    
    return fused_ext.fused_conv_transpose_postproc(
        x,
        conv_transpose_weight,
        conv_transpose_bias,
        bias,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        out_pad_d, out_pad_h, out_pad_w,
        conv_transpose_groups
    )

batch_size = 16
in_channels = 32
out_channels = 64
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1, 1)

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape]

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width).cuda()]
