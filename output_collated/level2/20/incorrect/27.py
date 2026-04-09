# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_130922/code_7.py
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

# Optimized CUDA kernel code with memory coalescing and reduced integer division
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Optimized kernel with memory coalescing and channel index optimization
__global__ void fused_post_conv_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int64_t num_elements,
    int64_t spatial_size,
    int64_t out_channels
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_elements) {
        // Optimized channel index calculation to reduce integer division
        int64_t channel_idx = (idx / spatial_size) % out_channels;
        
        float x = input[idx];
        float b = bias[channel_idx];
        
        // Compute: ((x + b) + x) * x + x = 2*x^2 + x*b + x
        output[idx] = (2.0f * x * x) + (x * b) + x;
    }
}

// Manual 3D transposed convolution kernel
__global__ void conv_transpose3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_d, int input_h, int input_w,
    int output_d, int output_h, int output_w,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int output_padding_d, int output_padding_h, int output_padding_w,
    int dilation_d, int dilation_h, int dilation_w
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output_elements = batch_size * out_channels * output_d * output_h * output_w;
    
    if (out_idx >= total_output_elements) return;
    
    // Calculate output indices
    int tmp = out_idx;
    int w_out = tmp % output_w; tmp /= output_w;
    int h_out = tmp % output_h; tmp /= output_h;
    int d_out = tmp % output_d; tmp /= output_d;
    int c_out = tmp % out_channels; tmp /= out_channels;
    int n = tmp;
    
    float sum = 0.0f;
    
    // Loop over input channels and kernel dimensions
    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int kd = 0; kd < kernel_d; ++kd) {
            for (int kh = 0; kh < kernel_h; ++kh) {
                for (int kw = 0; kw < kernel_w; ++kw) {
                    // Calculate corresponding input position
                    int d_in = d_out + padding_d - kd * dilation_d;
                    int h_in = h_out + padding_h - kh * dilation_h;
                    int w_in = w_out + padding_w - kw * dilation_w;
                    
                    // Check if within input bounds after accounting for stride
                    if (d_in % stride_d == 0 && h_in % stride_h == 0 && w_in % stride_w == 0) {
                        d_in /= stride_d;
                        h_in /= stride_h;
                        w_in /= stride_w;
                        
                        if (d_in >= 0 && d_in < input_d &&
                            h_in >= 0 && h_in < input_h &&
                            w_in >= 0 && w_in < input_w) {
                            
                            // Calculate indices
                            int input_idx = ((((n * in_channels + c_in) * input_d + d_in) * input_h + h_in) * input_w + w_in);
                            int weight_idx = (((((c_out * in_channels + c_in) * kernel_d + kd) * kernel_h + kh) * kernel_w + kw));
                            
                            sum += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
    }
    
    // Add bias
    sum += bias[c_out];
    
    // Write to output
    output[out_idx] = sum;
}

void fused_post_conv_forward(
    const torch::Tensor& input,
    const torch::Tensor& bias,
    torch::Tensor& output
) {
    int64_t num_elements = input.numel();
    int64_t spatial_size = input.size(2) * input.size(3) * input.size(4);
    int64_t out_channels = input.size(1);
    
    const int threads = 256;
    const int blocks = (num_elements + threads - 1) / threads;
    
    fused_post_conv_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        num_elements,
        spatial_size,
        out_channels
    );
}

void conv_transpose3d_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    int64_t groups,
    std::vector<int64_t> dilation
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_d = input.size(2);
    int input_h = input.size(3);
    int input_w = input.size(4);
    
    int out_channels = weight.size(1) * groups;  // Adjusted for groups
    int kernel_d = weight.size(2);
    int kernel_h = weight.size(3);
    int kernel_w = weight.size(4);
    
    int output_d = (input_d - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kernel_d - 1) + output_padding[0] + 1;
    int output_h = (input_h - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kernel_h - 1) + output_padding[1] + 1;
    int output_w = (input_w - 1) * stride[2] - 2 * padding[2] + dilation[2] * (kernel_w - 1) + output_padding[2] + 1;
    
    int total_output_elements = batch_size * out_channels * output_d * output_h * output_w;
    const int threads = 256;
    const int blocks = (total_output_elements + threads - 1) / threads;
    
    conv_transpose3d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_d, input_h, input_w,
        output_d, output_h, output_w,
        kernel_d, kernel_h, kernel_w,
        stride[0], stride[1], stride[2],
        padding[0], padding[1], padding[2],
        output_padding[0], output_padding[1], output_padding[2],
        dilation[0], dilation[1], dilation[2]
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
#include <vector>

// Forward declarations
void fused_post_conv_forward(const torch::Tensor& input, const torch::Tensor& bias, torch::Tensor& output);
void conv_transpose3d_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    int64_t groups,
    std::vector<int64_t> dilation
);

// Binding functions
torch::Tensor fused_post_conv(const torch::Tensor& input, const torch::Tensor& bias) {
    auto output = torch::empty_like(input);
    fused_post_conv_forward(input, bias, output);
    return output;
}

torch::Tensor manual_conv_transpose3d(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding,
    int64_t groups,
    std::vector<int64_t> dilation
) {
    // Calculate output dimensions
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_d = input.size(2);
    int input_h = input.size(3);
    int input_w = input.size(4);
    
    int out_channels = weight.size(1) * groups;
    int kernel_d = weight.size(2);
    int kernel_h = weight.size(3);
    int kernel_w = weight.size(4);
    
    int output_d = (input_d - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kernel_d - 1) + output_padding[0] + 1;
    int output_h = (input_h - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kernel_h - 1) + output_padding[1] + 1;
    int output_w = (input_w - 1) * stride[2] - 2 * padding[2] + dilation[2] * (kernel_w - 1) + output_padding[2] + 1;
    
    auto output = torch::empty({batch_size, out_channels, output_d, output_h, output_w}, 
                               torch::TensorOptions().dtype(input.dtype()).device(input.device()));
    
    conv_transpose3d_forward(input, weight, bias, output, stride, padding, output_padding, groups, dilation);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_post_conv", &fused_post_conv, "Fused post-convolution operation");
    m.def("manual_conv_transpose3d", &manual_conv_transpose3d, "Manual 3D transposed convolution");
}
"""

# Compile the extension
module = load_inline(
    name='optimized_fused_ops',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
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
    # Perform manual 3D transposed convolution
    x = module.manual_conv_transpose3d(
        x,
        conv_transpose_weight,
        conv_transpose_bias,
        conv_transpose_stride,
        conv_transpose_padding,
        conv_transpose_output_padding,
        conv_transpose_groups,
        conv_transpose_dilation
    )
    
    # Apply fused post-processing operation
    return module.fused_post_conv(x, bias.view(-1))

# Test parameters
batch_size = 16
in_channels = 32
out_channels = 64
depth, height, width = 16, 32, 32
kernel_size = 3
stride = [2, 2, 2]
padding = [1, 1, 1]
output_padding = [1, 1, 1]
bias_shape = (out_channels, 1, 1, 1)

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape]

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width).cuda()]
