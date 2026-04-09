# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_035600/code_0.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'output_padding', 'bias_shape']
FORWARD_ARG_NAMES = ['x', 'add_input']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'bias']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a 3D transposed convolution, adds an input tensor, and applies HardSwish activation.
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# CUDA kernel implementation
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_conv_transpose3d_add_hardswish_kernel(
    const float* input,
    const float* add_input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_d, int input_h, int input_w,
    int output_d, int output_h, int output_w,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int dilation_d, int dilation_h, int dilation_w,
    int groups) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * output_d * output_h * output_w;
    
    if (tid >= total_elements) return;
    
    // Calculate output indices
    int w_idx = tid % output_w;
    int h_idx = (tid / output_w) % output_h;
    int d_idx = (tid / (output_w * output_h)) % output_d;
    int c_idx = (tid / (output_w * output_h * output_d)) % out_channels;
    int b_idx = tid / (output_w * output_h * output_d * out_channels);
    
    float sum = (bias != nullptr) ? bias[c_idx] : 0.0f;
    
    // ConvTranspose3D computation
    int group_idx = c_idx * groups / out_channels;
    int in_channels_per_group = in_channels / groups;
    int out_channels_per_group = out_channels / groups;
    
    for (int kd = 0; kd < kernel_d; ++kd) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                // Calculate corresponding input position
                int input_d_idx = d_idx + padding_d - kd * dilation_d;
                int input_h_idx = h_idx + padding_h - kh * dilation_h;
                int input_w_idx = w_idx + padding_w - kw * dilation_w;
                
                // Check if this is a valid input position considering stride
                if (input_d_idx >= 0 && input_h_idx >= 0 && input_w_idx >= 0 &&
                    input_d_idx % stride_d == 0 && input_h_idx % stride_h == 0 && input_w_idx % stride_w == 0) {
                    
                    input_d_idx /= stride_d;
                    input_h_idx /= stride_h;
                    input_w_idx /= stride_w;
                    
                    // Check if the input position is within bounds
                    if (input_d_idx < input_d && input_h_idx < input_h && input_w_idx < input_w) {
                        // Process all input channels in this group
                        for (int ic = 0; ic < in_channels_per_group; ++ic) {
                            int input_channel = group_idx * in_channels_per_group + ic;
                            
                            int input_idx = b_idx * (in_channels * input_d * input_h * input_w) +
                                          input_channel * (input_d * input_h * input_w) +
                                          input_d_idx * (input_h * input_w) +
                                          input_h_idx * input_w +
                                          input_w_idx;
                                          
                            int weight_idx = c_idx * (in_channels_per_group * kernel_d * kernel_h * kernel_w) +
                                           ic * (kernel_d * kernel_h * kernel_w) +
                                           kd * (kernel_h * kernel_w) +
                                           kh * kernel_w +
                                           kw;
                            
                            sum += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
    }
    
    // Add input
    sum += add_input[tid];
    
    // Hardswish activation: x * relu6(x + 3) / 6
    float relu6_val = fminf(fmaxf(sum + 3.0f, 0.0f), 6.0f);
    output[tid] = sum * relu6_val / 6.0f;
}

void fused_conv_transpose3d_add_hardswish(
    const torch::Tensor& input,
    const torch::Tensor& add_input,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias,
    torch::Tensor& output,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int dilation_d, int dilation_h, int dilation_w,
    int groups) {
    
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_d = input.size(2);
    int input_h = input.size(3);
    int input_w = input.size(4);
    
    int out_channels = weight.size(0);
    int output_d = add_input.size(2);
    int output_h = add_input.size(3);
    int output_w = add_input.size(4);
    
    int total_elements = batch_size * out_channels * output_d * output_h * output_w;
    int threads_per_block = 256;
    int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    fused_conv_transpose3d_add_hardswish_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        add_input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.has_value() ? bias.value().data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_d, input_h, input_w,
        output_d, output_h, output_w,
        kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w,
        dilation_d, dilation_h, dilation_w,
        groups
    );
}
"""

# C++ binding code
cpp_source = r"""
#include <torch/extension.h>

void fused_conv_transpose3d_add_hardswish(
    const torch::Tensor& input,
    const torch::Tensor& add_input,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias,
    torch::Tensor& output,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int dilation_d, int dilation_h, int dilation_w,
    int groups);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_conv_transpose3d_add_hardswish, "Fused ConvTranspose3D + Add + Hardswish");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_conv_transpose_add_hardswish',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(
    x,
    add_input,
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
    # Ensure inputs are on CUDA
    if not x.is_cuda:
        x = x.cuda()
    if not add_input.is_cuda:
        add_input = add_input.cuda()
    if not conv_transpose_weight.is_cuda:
        conv_transpose_weight = conv_transpose_weight.cuda()
    if conv_transpose_bias is not None and not conv_transpose_bias.is_cuda:
        conv_transpose_bias = conv_transpose_bias.cuda()
    
    # Create output tensor
    output = torch.empty_like(add_input)
    
    # Extract kernel size from weight tensor
    kernel_d, kernel_h, kernel_w = conv_transpose_weight.shape[2], conv_transpose_weight.shape[3], conv_transpose_weight.shape[4]
    
    # Extract stride values
    if isinstance(conv_transpose_stride, int):
        stride_d = stride_h = stride_w = conv_transpose_stride
    else:
        stride_d, stride_h, stride_w = conv_transpose_stride[0], conv_transpose_stride[1], conv_transpose_stride[2]
    
    # Extract padding values
    if isinstance(conv_transpose_padding, int):
        padding_d = padding_h = padding_w = conv_transpose_padding
    else:
        padding_d, padding_h, padding_w = conv_transpose_padding[0], conv_transpose_padding[1], conv_transpose_padding[2]
    
    # Extract dilation values
    if isinstance(conv_transpose_dilation, int):
        dilation_d = dilation_h = dilation_w = conv_transpose_dilation
    else:
        dilation_d, dilation_h, dilation_w = conv_transpose_dilation[0], conv_transpose_dilation[1], conv_transpose_dilation[2]
    
    # Call fused operation
    fused_ext.fused_op(
        x, add_input, conv_transpose_weight, conv_transpose_bias, output,
        kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w,
        dilation_d, dilation_h, dilation_w,
        conv_transpose_groups
    )
    
    return output

batch_size = 128
in_channels = 32
out_channels = 64
D, H, W = 16, 16, 16
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1, 1, 1)

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape]

def get_inputs():
    return [torch.rand(batch_size, in_channels, D, H, W, device='cuda'), 
            torch.rand(batch_size, out_channels, D*stride, H*stride, W*stride, device='cuda')]
