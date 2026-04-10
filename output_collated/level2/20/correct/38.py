# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_140117/code_1.py
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
# Optimized CUDA kernels - fused transposed convolution + post processing
# -------------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// Device function for the fused post-processing operation
__device__ __forceinline__ float apply_fused_operation(float x, float bias_val) {
    // Compute ((x + bias) + x) * x + x  = 2*x*x + bias*x + x
    float tmp  = x + bias_val;          // x + b
    float tmp2 = tmp + x;               // (x + b) + x = 2*x + b
    float res  = tmp2 * x + x;          // ((2*x + b) * x) + x
    return res;
}

__global__ void fused_transpose_conv_post_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ post_bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int in_depth,
    const int in_height,
    const int in_width,
    const int out_depth,
    const int out_height,
    const int out_width,
    const int kernel_size,
    const int stride,
    const int padding,
    const int output_padding)
{
    // Calculate output indices
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output_elements = batch_size * out_channels * out_depth * out_height * out_width;
    
    if (out_idx >= total_output_elements) return;
    
    // Decompose linear index into multidimensional coordinates
    int tmp = out_idx;
    int w_out = tmp % out_width; tmp /= out_width;
    int h_out = tmp % out_height; tmp /= out_height;
    int d_out = tmp % out_depth; tmp /= out_depth;
    int c_out = tmp % out_channels; tmp /= out_channels;
    int n = tmp;
    
    // Calculate corresponding input region
    float sum = 0.0f;
    
    // Iterate through kernel positions that could contribute to this output
    for (int kd = 0; kd < kernel_size; ++kd) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                // Map output coordinates to input coordinates
                int d_in = (d_out + padding - kd);
                int h_in = (h_out + padding - kh);
                int w_in = (w_out + padding - kw);
                
                // Check if the input coordinate is valid after division by stride
                if (d_in >= 0 && d_in < in_depth * stride && (d_in % stride) == 0 &&
                    h_in >= 0 && h_in < in_height * stride && (h_in % stride) == 0 &&
                    w_in >= 0 && w_in < in_width * stride && (w_in % stride) == 0) {
                    
                    d_in /= stride;
                    h_in /= stride;
                    w_in /= stride;
                    
                    if (d_in < in_depth && h_in < in_height && w_in < in_width) {
                        // Accumulate for all input channels
                        for (int c_in = 0; c_in < in_channels; ++c_in) {
                            int input_idx = ((n * in_channels + c_in) * in_depth + d_in) * in_height * in_width +
                                            h_in * in_width + w_in;
                            
                            // Transposed convolution weight layout: [in_channels, out_channels/group, kD, kH, kW]
                            int weight_idx = ((c_in * out_channels + c_out) * kernel_size + kd) * kernel_size * kernel_size + 
                                             kh * kernel_size + kw;
                            
                            sum += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
    }
    
    // Add convolution bias
    sum += conv_bias[c_out];
    
    // Apply fused post-processing operation
    float bias_val = post_bias[c_out];
    float result = apply_fused_operation(sum, bias_val);
    
    output[out_idx] = result;
}

void fused_transpose_conv_post_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& conv_bias,
    const torch::Tensor& post_bias,
    torch::Tensor& output,
    int stride,
    int padding,
    int output_padding)
{
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_depth = input.size(2);
    int in_height = input.size(3);
    int in_width = input.size(4);
    
    int out_channels = weight.size(1);
    int kernel_size = weight.size(2);
    
    // Calculate output dimensions
    int out_depth = (in_depth - 1) * stride - 2 * padding + kernel_size + output_padding;
    int out_height = (in_height - 1) * stride - 2 * padding + kernel_size + output_padding;
    int out_width = (in_width - 1) * stride - 2 * padding + kernel_size + output_padding;
    
    int total_output_elements = batch_size * out_channels * out_depth * out_height * out_width;
    
    // Launch configuration
    const int threads_per_block = 256;
    const int blocks = (total_output_elements + threads_per_block - 1) / threads_per_block;
    
    fused_transpose_conv_post_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        conv_bias.data_ptr<float>(),
        post_bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        in_depth,
        in_height,
        in_width,
        out_depth,
        out_height,
        out_width,
        kernel_size,
        stride,
        padding,
        output_padding);
}
"""

# -------------------------------------------------------------------------
# C++ binding (PYBIND11)
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void fused_transpose_conv_post_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& conv_bias,
    const torch::Tensor& post_bias,
    torch::Tensor& output,
    int stride,
    int padding,
    int output_padding);

torch::Tensor fused_transpose_conv_post(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& conv_bias,
    const torch::Tensor& post_bias,
    int stride,
    int padding,
    int output_padding) {
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(conv_bias.is_contiguous(), "conv_bias must be contiguous");
    TORCH_CHECK(post_bias.is_contiguous(), "post_bias must be contiguous");
    
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_depth = input.size(2);
    int in_height = input.size(3);
    int in_width = input.size(4);
    
    int out_channels = weight.size(1);
    int kernel_size = weight.size(2);
    
    // Calculate output dimensions
    int out_depth = (in_depth - 1) * stride - 2 * padding + kernel_size + output_padding;
    int out_height = (in_height - 1) * stride - 2 * padding + kernel_size + output_padding;
    int out_width = (in_width - 1) * stride - 2 * padding + kernel_size + output_padding;
    
    auto output = torch::empty({batch_size, out_channels, out_depth, out_height, out_width}, 
                               torch::dtype(input.dtype()).device(input.device()));
    
    fused_transpose_conv_post_forward(input, weight, conv_bias, post_bias, output, stride, padding, output_padding);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_transpose_conv_post", &fused_transpose_conv_post, "Fused transposed convolution with post processing");
}
"""

# -------------------------------------------------------------------------
# Build the extension
# -------------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_transpose_conv_post_ext',
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
    
    # Use our fused kernel instead of separate conv_transpose3d + post-processing
    return fused_ext.fused_transpose_conv_post(
        x,
        conv_transpose_weight,
        conv_transpose_bias,
        bias_flat,
        conv_transpose_stride[0],  # Assuming uniform stride
        conv_transpose_padding[0],  # Assuming uniform padding
        conv_transpose_output_padding[0]  # Assuming uniform output padding
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
