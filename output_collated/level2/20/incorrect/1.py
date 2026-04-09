# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_124936/code_0.py
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void conv_transpose3d_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_depth,
    int input_height,
    int input_width,
    int output_depth,
    int output_height,
    int output_width,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    int groups,
    int dilation
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * output_depth * output_height * output_width;
    
    if (out_idx >= total_elements) return;
    
    int w = out_idx % output_width;
    out_idx /= output_width;
    int h = out_idx % output_height;
    out_idx /= output_height;
    int d = out_idx % output_depth;
    out_idx /= output_depth;
    int out_ch = out_idx % out_channels;
    int batch = out_idx / out_channels;
    
    float sum = 0.0f;
    
    // Calculate input indices that contribute to this output element
    int group = out_ch * groups / out_channels;
    int weight_offset_base = (out_ch * in_channels / groups + group * in_channels / groups) * 
                             kernel_size * kernel_size * kernel_size;
    
    for (int kd = 0; kd < kernel_size; kd++) {
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                // Calculate corresponding input coordinates
                int in_d = d + padding - kd * dilation;
                int in_h = h + padding - kh * dilation;
                int in_w = w + padding - kw * dilation;
                
                // Check if indices are valid after accounting for stride
                if (in_d % stride == 0 && in_h % stride == 0 && in_w % stride == 0) {
                    in_d /= stride;
                    in_h /= stride;
                    in_w /= stride;
                    
                    if (in_d >= 0 && in_d < input_depth && 
                        in_h >= 0 && in_h < input_height && 
                        in_w >= 0 && in_w < input_width) {
                        
                        // Calculate input index
                        int in_ch_start = group * (in_channels / groups);
                        int in_ch_end = (group + 1) * (in_channels / groups);
                        
                        for (int in_ch = in_ch_start; in_ch < in_ch_end; in_ch++) {
                            int input_idx = ((batch * in_channels + in_ch) * input_depth + in_d) * 
                                            input_height * input_width + in_h * input_width + in_w;
                            
                            int weight_idx = weight_offset_base + 
                                            ((in_ch - in_ch_start) * kernel_size + kd) * kernel_size * kernel_size + 
                                            kh * kernel_size + kw;
                            
                            sum += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
    }
    
    // Add bias if provided
    if (bias != nullptr) {
        sum += bias[out_ch];
    }
    
    // Apply fused operations: ((x + bias_add) + x) * x + x
    float x_val = sum;
    float bias_add = ((float*)bias)[out_ch]; // Using out_ch to index into bias tensor
    float biased_val = x_val + bias_add;
    float result = (biased_val + x_val) * x_val + x_val;
    
    output[out_idx * output_depth * output_height * output_width + 
           d * output_height * output_width + 
           h * output_width + 
           w] = result;
}

__global__ void fused_post_conv_kernel(
    const float* input,
    const float* bias,
    float* output,
    int batch_size,
    int channels,
    int depth,
    int height,
    int width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * depth * height * width;
    
    if (idx < total_elements) {
        float x_val = input[idx];
        int channel_idx = (idx / (depth * height * width)) % channels;
        float bias_val = bias[channel_idx];
        
        // Compute: ((x + bias) + x) * x + x
        float result = ((x_val + bias_val) + x_val) * x_val + x_val;
        output[idx] = result;
    }
}

void conv_transpose3d_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& conv_bias,
    const torch::Tensor& post_bias,
    torch::Tensor& output,
    int stride,
    int padding,
    int output_padding,
    int groups,
    int dilation
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_depth = input.size(2);
    int input_height = input.size(3);
    int input_width = input.size(4);
    
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2); // Assuming cubic kernel
    
    int output_depth = (input_depth - 1) * stride - 2 * padding + kernel_size + output_padding;
    int output_height = (input_height - 1) * stride - 2 * padding + kernel_size + output_padding;
    int output_width = (input_width - 1) * stride - 2 * padding + kernel_size + output_padding;
    
    int total_elements = batch_size * out_channels * output_depth * output_height * output_width;
    int threads_per_block = 256;
    int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    conv_transpose3d_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        conv_bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_depth,
        input_height,
        input_width,
        output_depth,
        output_height,
        output_width,
        kernel_size,
        stride,
        padding,
        output_padding,
        groups,
        dilation
    );
}

void fused_post_conv_forward(
    const torch::Tensor& input,
    const torch::Tensor& bias,
    torch::Tensor& output
) {
    int batch_size = input.size(0);
    int channels = input.size(1);
    int depth = input.size(2);
    int height = input.size(3);
    int width = input.size(4);
    
    int total_elements = batch_size * channels * depth * height * width;
    int threads_per_block = 256;
    int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    fused_post_conv_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        depth,
        height,
        width
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void conv_transpose3d_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& conv_bias,
    const torch::Tensor& post_bias,
    torch::Tensor& output,
    int stride,
    int padding,
    int output_padding,
    int groups,
    int dilation
);

void fused_post_conv_forward(
    const torch::Tensor& input,
    const torch::Tensor& bias,
    torch::Tensor& output
);

torch::Tensor custom_conv_transpose3d(const torch::Tensor& input,
                                      const torch::Tensor& weight,
                                      const torch::Tensor& conv_bias,
                                      const torch::Tensor& post_bias,
                                      int stride,
                                      int padding,
                                      int output_padding,
                                      int groups,
                                      int dilation) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_depth = input.size(2);
    int input_height = input.size(3);
    int input_width = input.size(4);
    
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);
    
    int output_depth = (input_depth - 1) * stride - 2 * padding + kernel_size + output_padding;
    int output_height = (input_height - 1) * stride - 2 * padding + kernel_size + output_padding;
    int output_width = (input_width - 1) * stride - 2 * padding + kernel_size + output_padding;
    
    auto output = torch::empty({batch_size, out_channels, output_depth, output_height, output_width}, 
                               torch::dtype(input.dtype()).device(input.device()));
    
    conv_transpose3d_forward(input, weight, conv_bias, post_bias, output,
                             stride, padding, output_padding, groups, dilation);
    return output;
}

torch::Tensor fused_post_conv(const torch::Tensor& input, const torch::Tensor& bias) {
    auto output = torch::empty_like(input);
    fused_post_conv_forward(input, bias, output);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("custom_conv_transpose3d", &custom_conv_transpose3d, "Custom ConvTranspose3D with fused operations");
    m.def("fused_post_conv", &fused_post_conv, "Fused post-convolution operations");
}
"""

fused_ext = load_inline(
    name='fused_conv_ops',
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
    # Use custom CUDA implementation for the entire operation
    return fused_ext.custom_conv_transpose3d(
        x, 
        conv_transpose_weight, 
        conv_transpose_bias, 
        bias,
        conv_transpose_stride, 
        conv_transpose_padding, 
        conv_transpose_output_padding, 
        conv_transpose_groups, 
        conv_transpose_dilation
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
    return [torch.rand(batch_size, in_channels, depth, height, width)]
