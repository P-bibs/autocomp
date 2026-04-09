# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_121229/code_11.py
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused conv_transpose2d + subtraction + tanh
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_conv_transpose_sub_tanh_kernel(
    const float* x,
    const float* weight,
    const float* conv_bias,
    const float* sub_bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int x_h,
    int x_w,
    int weight_h,
    int weight_w,
    int out_h,
    int out_w,
    int stride,
    int padding,
    int output_padding,
    int groups,
    int dilation
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * out_h * out_w;
    
    if (idx >= total_elements) return;
    
    // Decompose linear index
    int remaining = idx;
    int w = remaining % out_w;
    remaining /= out_w;
    int h = remaining % out_h;
    remaining /= out_h;
    int oc = remaining % out_channels;
    remaining /= out_channels;
    int b = remaining;
    
    // Compute convolution output value
    float conv_out = 0.0f;
    
    if (conv_bias != nullptr) {
        conv_out = conv_bias[oc];
    }
    
    int group_in_channels = in_channels / groups;
    int group_out_channels = out_channels / groups;
    int group_idx = oc / group_out_channels;
    
    for (int kh = 0; kh < weight_h; kh++) {
        for (int kw = 0; kw < weight_w; kw++) {
            int x_h_start = h - (weight_h - 1 - kh) * dilation + padding;
            int x_w_start = w - (weight_w - 1 - kw) * dilation + padding;
            
            if (x_h_start % stride == 0 && x_w_start % stride == 0) {
                int x_h_idx = x_h_start / stride;
                int x_w_idx = x_w_start / stride;
                
                if (x_h_idx >= 0 && x_h_idx < x_h && x_w_idx >= 0 && x_w_idx < x_w) {
                    for (int ic = 0; ic < group_in_channels; ic++) {
                        int in_ch = group_idx * group_in_channels + ic;
                        int x_idx = ((b * in_channels + in_ch) * x_h + x_h_idx) * x_w + x_w_idx;
                        int w_idx = ((oc * group_in_channels + ic) * weight_h + kh) * weight_w + kw;
                        conv_out += x[x_idx] * weight[w_idx];
                    }
                }
            }
        }
    }
    
    // Fused: subtract bias and apply tanh
    float sub_bias_val = 0.0f;
    if (sub_bias != nullptr) {
        sub_bias_val = sub_bias[oc];
    }
    
    float result = conv_out - sub_bias_val;
    output[idx] = tanhf(result);
}

void fused_conv_transpose_sub_tanh_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor conv_bias,
    torch::Tensor sub_bias,
    torch::Tensor output,
    int stride,
    int padding,
    int output_padding,
    int groups,
    int dilation
) {
    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int out_channels = weight.size(1);
    int x_h = x.size(2);
    int x_w = x.size(3);
    int weight_h = weight.size(2);
    int weight_w = weight.size(3);
    int out_h = output.size(2);
    int out_w = output.size(3);
    
    const float* x_ptr = x.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* conv_bias_ptr = conv_bias.defined() ? conv_bias.data_ptr<float>() : nullptr;
    const float* sub_bias_ptr = sub_bias.defined() ? sub_bias.data_ptr<float>() : nullptr;
    float* output_ptr = output.data_ptr<float>();
    
    int total_elements = batch_size * out_channels * out_h * out_w;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    
    fused_conv_transpose_sub_tanh_kernel<<<blocks, threads>>>(
        x_ptr, weight_ptr, conv_bias_ptr, sub_bias_ptr, output_ptr,
        batch_size, in_channels, out_channels, x_h, x_w,
        weight_h, weight_w, out_h, out_w,
        stride, padding, output_padding, groups, dilation
    );
    
    cudaDeviceSynchronize();
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_conv_transpose_sub_tanh_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor conv_bias,
    torch::Tensor sub_bias,
    torch::Tensor output,
    int stride,
    int padding,
    int output_padding,
    int groups,
    int dilation
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_transpose_sub_tanh", &fused_conv_transpose_sub_tanh_forward, 
          "Fused conv_transpose2d + subtraction + tanh kernel");
}
"""

fused_ext = load_inline(
    name='fused_conv_transpose_sub_tanh',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True,
    verbose=True
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
    # Compute output shape
    batch_size, in_channels, x_h, x_w = x.shape
    out_channels, _, weight_h, weight_w = conv_transpose_weight.shape
    
    out_h = (x_h - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_dilation * (weight_h - 1) + conv_transpose_output_padding + 1
    out_w = (x_w - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_dilation * (weight_w - 1) + conv_transpose_output_padding + 1
    
    # Allocate output tensor
    output = torch.zeros(batch_size, out_channels, out_h, out_w, dtype=x.dtype, device=x.device)
    
    # Call fused kernel
    fused_ext.fused_conv_transpose_sub_tanh(
        x,
        conv_transpose_weight,
        conv_transpose_bias,
        bias,
        output,
        conv_transpose_stride,
        conv_transpose_padding,
        conv_transpose_output_padding,
        conv_transpose_groups,
        conv_transpose_dilation
    )
    
    return output


# Test parameters
batch_size = 32
in_channels = 64
out_channels = 64
height = width = 256
kernel_size = 4
bias_shape = (out_channels, 1, 1)

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]
