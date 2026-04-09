# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_133559/code_8.py
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
# Optimized CUDA kernel – Fused ConvTranspose3d + Post-processing
# -------------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define CONV3D_MAX_KERNEL_SIZE 5

__global__ void fused_conv_transpose3d_post_process_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ post_bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int in_depth,
    const int in_height,
    const int in_width,
    const int out_channels,
    const int out_depth,
    const int out_height,
    const int out_width,
    const int kernel_d,
    const int kernel_h,
    const int kernel_w,
    const int stride_d,
    const int stride_h,
    const int stride_w,
    const int padding_d,
    const int padding_h,
    const int padding_w,
    const int output_padding_d,
    const int output_padding_h,
    const int output_padding_w,
    const int dilation_d,
    const int dilation_h,
    const int dilation_w
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * out_channels * out_depth * out_height * out_width;

    if (out_idx >= total_elements) return;

    // Compute output indices
    const int w_out = out_idx % out_width;
    const int h_out = (out_idx / out_width) % out_height;
    const int d_out = (out_idx / (out_width * out_height)) % out_depth;
    const int c_out = (out_idx / (out_width * out_height * out_depth)) % out_channels;
    const int n_out = out_idx / (out_width * out_height * out_depth * out_channels);

    float value = 0.0f;

    // Iterate over kernel dimensions
    for (int kd = 0; kd < kernel_d; ++kd) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                // Calculate corresponding input position
                int d_in = d_out + padding_d - kd * dilation_d;
                int h_in = h_out + padding_h - kh * dilation_h;
                int w_in = w_out + padding_w - kw * dilation_w;

                // Check if within input bounds and stride alignment
                if (d_in % stride_d == 0 && h_in % stride_h == 0 && w_in % stride_w == 0) {
                    d_in /= stride_d;
                    h_in /= stride_h;
                    w_in /= stride_w;
                    
                    if (d_in >= 0 && d_in < in_depth &&
                        h_in >= 0 && h_in < in_height &&
                        w_in >= 0 && w_in < in_width) {
                        
                        // Compute input index
                        const int in_idx = ((((n_out * in_channels + 0) * in_depth + d_in) * in_height + h_in) * in_width + w_in);
                        const int weight_idx = (((((c_out * in_channels) + 0) * kernel_d + kd) * kernel_h + kh) * kernel_w + kw);
                        
                        for (int c_in = 0; c_in < in_channels; ++c_in) {
                            const int in_idx_c = in_idx + c_in * in_depth * in_height * in_width;
                            const int weight_idx_c = weight_idx + c_in * kernel_d * kernel_h * kernel_w;
                            value += input[in_idx_c] * weight[weight_idx_c];
                        }
                    }
                }
            }
        }
    }

    // Add convolution bias
    value += conv_bias[c_out];

    // Apply post-processing: ((x + bias) + x) * x + x = 2*x*x + bias*x + x
    const int post_bias_idx = c_out;
    const float post_bias_val = post_bias[post_bias_idx];
    
    const float tmp  = value + post_bias_val;     // x + b
    const float tmp2 = tmp + value;               // (x + b) + x = 2*x + b
    const float result = tmp2 * value + value;    // ((2*x + b) * x) + x

    output[out_idx] = result;
}

void fused_conv_transpose3d_post_process_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& conv_bias,
    const torch::Tensor& post_bias,
    torch::Tensor& output,
    const int stride_d,
    const int stride_h,
    const int stride_w,
    const int padding_d,
    const int padding_h,
    const int padding_w,
    const int output_padding_d,
    const int output_padding_h,
    const int output_padding_w,
    const int dilation_d,
    const int dilation_h,
    const int dilation_w)
{
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int in_depth = input.size(2);
    const int in_height = input.size(3);
    const int in_width = input.size(4);

    const int out_channels = weight.size(1); // Note: For ConvTranspose, weight is [in_channels, out_channels, ...]
    const int kernel_d = weight.size(2);
    const int kernel_h = weight.size(3);
    const int kernel_w = weight.size(4);
    
    const int out_depth = (in_depth - 1) * stride_d - 2 * padding_d + dilation_d * (kernel_d - 1) + output_padding_d + 1;
    const int out_height = (in_height - 1) * stride_h - 2 * padding_h + dilation_h * (kernel_h - 1) + output_padding_h + 1;
    const int out_width = (in_width - 1) * stride_w - 2 * padding_w + dilation_w * (kernel_w - 1) + output_padding_w + 1;

    const int total_elements = batch_size * out_channels * out_depth * out_height * out_width;

    const int threads_per_block = 256;
    const int blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    fused_conv_transpose3d_post_process_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        conv_bias.data_ptr<float>(),
        post_bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        in_depth,
        in_height,
        in_width,
        out_channels,
        out_depth,
        out_height,
        out_width,
        kernel_d,
        kernel_h,
        kernel_w,
        stride_d,
        stride_h,
        stride_w,
        padding_d,
        padding_h,
        padding_w,
        output_padding_d,
        output_padding_h,
        output_padding_w,
        dilation_d,
        dilation_h,
        dilation_w
    );
}
"""

# -------------------------------------------------------------------------
# C++ binding (PYBIND11)
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void fused_conv_transpose3d_post_process_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& conv_bias,
    const torch::Tensor& post_bias,
    torch::Tensor& output,
    const int stride_d,
    const int stride_h,
    const int stride_w,
    const int padding_d,
    const int padding_h,
    const int padding_w,
    const int output_padding_d,
    const int output_padding_h,
    const int output_padding_w,
    const int dilation_d,
    const int dilation_h,
    const int dilation_w);

torch::Tensor fused_conv_transpose3d_post_process(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& conv_bias,
    const torch::Tensor& post_bias,
    const int stride_d,
    const int stride_h,
    const int stride_w,
    const int padding_d,
    const int padding_h,
    const int padding_w,
    const int output_padding_d,
    const int output_padding_h,
    const int output_padding_w,
    const int dilation_d,
    const int dilation_h,
    const int dilation_w) {
    
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(conv_bias.is_contiguous(), "conv_bias must be contiguous");
    TORCH_CHECK(post_bias.is_contiguous(), "post_bias must be contiguous");

    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int in_depth = input.size(2);
    const int in_height = input.size(3);
    const int in_width = input.size(4);

    const int out_channels = weight.size(1);
    const int kernel_d = weight.size(2);
    const int kernel_h = weight.size(3);
    const int kernel_w = weight.size(4);

    const int out_depth = (in_depth - 1) * stride_d - 2 * padding_d + dilation_d * (kernel_d - 1) + output_padding_d + 1;
    const int out_height = (in_height - 1) * stride_h - 2 * padding_h + dilation_h * (kernel_h - 1) + output_padding_h + 1;
    const int out_width = (in_width - 1) * stride_w - 2 * padding_w + dilation_w * (kernel_w - 1) + output_padding_w + 1;

    auto output = torch::empty({batch_size, out_channels, out_depth, out_height, out_width}, 
                               torch::TensorOptions().dtype(input.dtype()).device(input.device()));

    fused_conv_transpose3d_post_process_forward(
        input, weight, conv_bias, post_bias, output,
        stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w,
        output_padding_d, output_padding_h, output_padding_w,
        dilation_d, dilation_h, dilation_w
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_transpose3d_post_process", &fused_conv_transpose3d_post_process, 
          "Fused ConvTranspose3D with post-processing");
}
"""

# -------------------------------------------------------------------------
# Build the extension
# -------------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_conv_transpose3d_post_process_ext',
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
    # Extract stride components
    if isinstance(conv_transpose_stride, int):
        stride_d = stride_h = stride_w = conv_transpose_stride
    else:
        stride_d, stride_h, stride_w = conv_transpose_stride

    # Extract padding components
    if isinstance(conv_transpose_padding, int):
        padding_d = padding_h = padding_w = conv_transpose_padding
    else:
        padding_d, padding_h, padding_w = conv_transpose_padding

    # Extract output padding components
    if isinstance(conv_transpose_output_padding, int):
        output_padding_d = output_padding_h = output_padding_w = conv_transpose_output_padding
    else:
        output_padding_d, output_padding_h, output_padding_w = conv_transpose_output_padding

    # Extract dilation components
    if isinstance(conv_transpose_dilation, int):
        dilation_d = dilation_h = dilation_w = conv_transpose_dilation
    else:
        dilation_d, dilation_h, dilation_w = conv_transpose_dilation

    # Flatten bias to a 1-D tensor (required by the kernel)
    bias_flat = bias.view(-1)

    # Call the fused kernel
    return fused_ext.fused_conv_transpose3d_post_process(
        x,
        conv_transpose_weight,
        conv_transpose_bias,
        bias_flat,
        stride_d,
        stride_h,
        stride_w,
        padding_d,
        padding_h,
        padding_w,
        output_padding_d,
        output_padding_h,
        output_padding_w,
        dilation_d,
        dilation_h,
        dilation_w
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
