# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_132207/code_8.py
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

# -------------------------------------------------------------------------
# Optimized CUDA kernel – fused conv_transpose3d + post-processing
# -------------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_DIM 32
#define BLOCK_SIZE 16

__global__ void fused_conv_transpose3d_post_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ post_bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int in_depth, const int in_height, const int in_width,
    const int out_depth, const int out_height, const int out_width,
    const int kernel_size,
    const int stride, const int padding, const int output_padding,
    const int spatial_size_in,
    const int spatial_size_out)
{
    // Each thread handles one output element
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * out_depth * out_height * out_width) return;

    // Calculate indices
    int tmp = idx;
    const int w_out = tmp % out_width; tmp /= out_width;
    const int h_out = tmp % out_height; tmp /= out_height;
    const int d_out = tmp % out_depth; tmp /= out_depth;
    const int c_out = tmp % out_channels; tmp /= out_channels;
    const int b = tmp;

    // Compute convolution value
    float conv_val = 0.0f;
    
    // Iterate through kernel
    for (int kd = 0; kd < kernel_size; ++kd) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                // Map output coordinate to input coordinate
                const int d_in = (d_out + padding - kd) / stride;
                const int h_in = (h_out + padding - kh) / stride;
                const int w_in = (w_out + padding - kw) / stride;
                
                // Check bounds and divisibility
                if (d_in >= 0 && d_in < in_depth &&
                    h_in >= 0 && h_in < in_height &&
                    w_in >= 0 && w_in < in_width &&
                    (d_out + padding - kd) % stride == 0 &&
                    (h_out + padding - kh) % stride == 0 &&
                    (w_out + padding - kw) % stride == 0) {
                    
                    // Weight index: [out_channel, in_channel, kD, kH, kW]
                    const int w_idx = c_out * (in_channels * kernel_size * kernel_size * kernel_size) +
                                      ((kd * kernel_size + kh) * kernel_size + kw) * in_channels;
                    
                    // Input index: [batch, channel, depth, height, width]
                    const int i_idx = ((b * in_channels + 0) * in_depth + d_in) * in_height * in_width +
                                      h_in * in_width + w_in;
                    
                    // Accumulate across input channels
                    for (int c_in = 0; c_in < in_channels; ++c_in) {
                        const float w_val = weight[w_idx + c_in];
                        const float i_val = input[i_idx + c_in * spatial_size_in];
                        conv_val += w_val * i_val;
                    }
                }
            }
        }
    }

    // Add convolution bias
    conv_val += conv_bias[c_out];

    // Apply post-processing: ((x + bias) + x) * x + x = (2*x + bias) * x + x
    const int post_bias_idx = c_out;
    const float post_bias_val = post_bias[post_bias_idx];
    
    const float tmp1 = conv_val + post_bias_val;  // x + b
    const float tmp2 = tmp1 + conv_val;           // (x + b) + x = 2*x + b
    const float result = tmp2 * conv_val + conv_val;  // ((2*x + b) * x) + x

    // Write final result
    output[idx] = result;
}

void fused_conv_transpose3d_post_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& conv_bias,
    const torch::Tensor& post_bias,
    torch::Tensor& output,
    const int kernel_size,
    const int stride,
    const int padding,
    const int output_padding)
{
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int in_depth = input.size(2);
    const int in_height = input.size(3);
    const int in_width = input.size(4);
    
    const int out_channels = weight.size(0);
    const int out_depth = (in_depth - 1) * stride - 2 * padding + kernel_size + output_padding;
    const int out_height = (in_height - 1) * stride - 2 * padding + kernel_size + output_padding;
    const int out_width = (in_width - 1) * stride - 2 * padding + kernel_size + output_padding;
    
    const int spatial_size_in = in_depth * in_height * in_width;
    const int spatial_size_out = out_depth * out_height * out_width;
    const int total_elements = batch_size * out_channels * out_depth * out_height * out_width;

    const int threads_per_block = 256;
    const int blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    fused_conv_transpose3d_post_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        conv_bias.data_ptr<float>(),
        post_bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        in_depth, in_height, in_width,
        out_depth, out_height, out_width,
        kernel_size,
        stride, padding, output_padding,
        spatial_size_in,
        spatial_size_out);
}
"""

# -------------------------------------------------------------------------
# C++ binding (PYBIND11)
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void fused_conv_transpose3d_post_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& conv_bias,
    const torch::Tensor& post_bias,
    torch::Tensor& output,
    const int kernel_size,
    const int stride,
    const int padding,
    const int output_padding);

torch::Tensor fused_conv_transpose3d_post(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& conv_bias,
    const torch::Tensor& post_bias,
    const int kernel_size,
    const int stride,
    const int padding,
    const int output_padding) {
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(conv_bias.is_contiguous(), "conv_bias must be contiguous");
    TORCH_CHECK(post_bias.is_contiguous(), "post_bias must be contiguous");
    
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int in_depth = input.size(2);
    const int in_height = input.size(3);
    const int in_width = input.size(4);
    
    const int out_depth = (in_depth - 1) * stride - 2 * padding + kernel_size + output_padding;
    const int out_height = (in_height - 1) * stride - 2 * padding + kernel_size + output_padding;
    const int out_width = (in_width - 1) * stride - 2 * padding + kernel_size + output_padding;
    
    auto output = torch::empty({batch_size, weight.size(0), out_depth, out_height, out_width}, 
                               torch::dtype(input.dtype()).device(input.device()));
    
    fused_conv_transpose3d_post_forward(input, weight, conv_bias, post_bias, output,
                                        kernel_size, stride, padding, output_padding);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_transpose3d_post", &fused_conv_transpose3d_post, "Fused ConvTranspose3d and post-processing");
}
"""

# -------------------------------------------------------------------------
# Build the extension
# -------------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_conv_transpose3d_post_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# -------------------------------------------------------------------------
# Functional model used for evaluation
# -------------------------------------------------------------------------
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
    # Flatten bias to a 1-D tensor (required by the kernel)
    bias_flat = bias.view(-1)
    
    # Call fused kernel directly
    return fused_ext.fused_conv_transpose3d_post(
        x,
        conv_transpose_weight,
        conv_transpose_bias,
        bias_flat,
        conv_transpose_weight.size(2),  # kernel_size (assuming cubic)
        conv_transpose_stride[0],       # stride (assuming uniform)
        conv_transpose_padding[0],      # padding (assuming uniform)
        conv_transpose_output_padding[0] # output_padding (assuming uniform)
    )

# -------------------------------------------------------------------------
# Helper code (shape parameters, input factories)
# -------------------------------------------------------------------------
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
