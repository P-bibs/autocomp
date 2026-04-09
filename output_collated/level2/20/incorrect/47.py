# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_133559/code_4.py
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
from torch.utils.cpp_extension import load_inline

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_post_conv_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int num_elements,
    const int spatial_size,
    const int out_channels)
{
    // Grid-stride loop implementation
    int elements_in_grid = blockDim.x * gridDim.x * 4;
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    for (int i = idx; i < num_elements; i += elements_in_grid) {
        int channel_idx = (i / spatial_size) % out_channels;
        float bias_val = __ldg(&bias[channel_idx]);
        
        float4 result_vec;
        float* res_ptr = reinterpret_cast<float*>(&result_vec);
        
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            if (i + j < num_elements) {
                float x = __ldg(&input[i + j]);
                // Compute ((x + bias) + x) * x + x  = 2*x*x + bias*x + x
                float tmp  = x + bias_val;          // x + b
                float tmp2 = tmp + x;               // (x + b) + x = 2*x + b
                float res  = tmp2 * x + x;          // ((2*x + b) * x) + x
                res_ptr[j] = res;
            } else {
                res_ptr[j] = 0.0f;
            }
        }
        reinterpret_cast<float4*>(output + i)[0] = result_vec;
    }
}

__global__ void conv_transpose3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int input_depth,
    const int input_height,
    const int input_width,
    const int output_depth,
    const int output_height,
    const int output_width,
    const int kernel_size,
    const int stride,
    const int padding,
    const int output_padding)
{
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = batch_size * out_channels * output_depth * output_height * output_width;
    
    if (out_idx >= total_outputs) return;
    
    int w = out_idx % output_width;
    out_idx /= output_width;
    int h = out_idx % output_height;
    out_idx /= output_height;
    int d = out_idx % output_depth;
    out_idx /= output_depth;
    int c_out = out_idx % out_channels;
    int n = out_idx / out_channels;
    
    float sum = 0.0f;
    
    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int kd = 0; kd < kernel_size; ++kd) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    int in_d = d - stride * kd + padding;
                    int in_h = h - stride * kh + padding;
                    int in_w = w - stride * kw + padding;
                    
                    if (in_d >= 0 && in_d < input_depth * stride &&
                        in_h >= 0 && in_h < input_height * stride &&
                        in_w >= 0 && in_w < input_width * stride) {
                        
                        // Check if aligned with stride
                        if (in_d % stride == 0 && in_h % stride == 0 && in_w % stride == 0) {
                            in_d /= stride;
                            in_h /= stride;
                            in_w /= stride;
                            
                            if (in_d < input_depth && in_h < input_height && in_w < input_width) {
                                int input_idx = n * (in_channels * input_depth * input_height * input_width) +
                                                c_in * (input_depth * input_height * input_width) +
                                                in_d * (input_height * input_width) +
                                                in_h * input_width +
                                                in_w;
                                                
                                int weight_idx = c_in * (out_channels * kernel_size * kernel_size * kernel_size) +
                                                 c_out * (kernel_size * kernel_size * kernel_size) +
                                                 kd * (kernel_size * kernel_size) +
                                                 kh * kernel_size +
                                                 kw;
                                                 
                                sum += input[input_idx] * weight[weight_idx];
                            }
                        }
                    }
                }
            }
        }
    }
    
    output[out_idx * (out_channels * output_depth * output_height * output_width) +
           c_out * (output_depth * output_height * output_width) +
           d * (output_height * output_width) +
           h * output_width +
           w] = sum + bias[c_out];
}

void fused_post_conv_forward(const torch::Tensor& input, const torch::Tensor& bias, torch::Tensor& output) {
    const int num_elements = static_cast<int>(input.numel());
    const int spatial_size = static_cast<int>(input.size(2) * input.size(3) * input.size(4));
    const int out_channels = static_cast<int>(input.size(1));
    
    // Launch a fixed grid size regardless of tensor size
    const int threads = 256;
    const int blocks = 1024; 
    
    fused_post_conv_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(),
        num_elements, spatial_size, out_channels);
}

void conv_transpose3d_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    int stride,
    int padding,
    int output_padding)
{
    const int batch_size = static_cast<int>(input.size(0));
    const int in_channels = static_cast<int>(input.size(1));
    const int input_depth = static_cast<int>(input.size(2));
    const int input_height = static_cast<int>(input.size(3));
    const int input_width = static_cast<int>(input.size(4));
    
    const int out_channels = static_cast<int>(weight.size(1));
    const int kernel_size = static_cast<int>(weight.size(2));
    
    const int output_depth = (input_depth - 1) * stride - 2 * padding + kernel_size + output_padding;
    const int output_height = (input_height - 1) * stride - 2 * padding + kernel_size + output_padding;
    const int output_width = (input_width - 1) * stride - 2 * padding + kernel_size + output_padding;
    
    const int threads = 256;
    const int total_outputs = batch_size * out_channels * output_depth * output_height * output_width;
    const int blocks = (total_outputs + threads - 1) / threads;
    
    conv_transpose3d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
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
        output_padding);
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_post_conv_forward(const torch::Tensor& input, const torch::Tensor& bias, torch::Tensor& output);
void conv_transpose3d_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    int stride,
    int padding,
    int output_padding);

torch::Tensor fused_post_conv(const torch::Tensor& input, const torch::Tensor& bias) {
    auto output = torch::empty_like(input);
    fused_post_conv_forward(input, bias, output);
    return output;
}

torch::Tensor custom_conv_transpose3d(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    int stride,
    int padding,
    int output_padding) {
    
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int input_depth = input.size(2);
    const int input_height = input.size(3);
    const int input_width = input.size(4);
    
    const int out_channels = weight.size(1);
    const int kernel_size = weight.size(2);
    
    const int output_depth = (input_depth - 1) * stride - 2 * padding + kernel_size + output_padding;
    const int output_height = (input_height - 1) * stride - 2 * padding + kernel_size + output_padding;
    const int output_width = (input_width - 1) * stride - 2 * padding + kernel_size + output_padding;
    
    auto output = torch::empty({batch_size, out_channels, output_depth, output_height, output_width}, 
                               torch::dtype(input.dtype()).device(input.device()));
    
    conv_transpose3d_forward(input, weight, bias, output, stride, padding, output_padding);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_post_conv", &fused_post_conv, "Fused grid-stride kernel");
    m.def("custom_conv_transpose3d", &custom_conv_transpose3d, "Custom conv transpose 3d");
}
"""

fused_ext = load_inline(
    name='fused_post_conv_ext', cpp_sources=cpp_source, cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True
)

def functional_model(x, *, conv_transpose_weight, conv_transpose_bias, 
                     conv_transpose_stride, conv_transpose_padding, 
                     conv_transpose_output_padding, conv_transpose_groups, 
                     conv_transpose_dilation, bias):
    # Replace PyTorch's conv_transpose3d with our custom CUDA implementation
    x = fused_ext.custom_conv_transpose3d(
        x.contiguous(), 
        conv_transpose_weight.contiguous(), 
        conv_transpose_bias.contiguous(),
        conv_transpose_stride,
        conv_transpose_padding,
        conv_transpose_output_padding
    )
    
    # Apply the optimized fused post-processing kernel
    return fused_ext.fused_post_conv(x.contiguous(), bias.view(-1).contiguous())
