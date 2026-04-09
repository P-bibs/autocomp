# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_132207/code_4.py
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

# -------------------------------------------------------------------------
# Optimized CUDA kernel using Grid-Stride Loops
# -------------------------------------------------------------------------
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
    // Grid-Stride Loop: Each thread processes multiple elements
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int idx = tid; idx < num_elements; idx += stride) {
        int channel_idx = (idx / spatial_size) % out_channels;
        float x = input[idx];
        float b = bias[channel_idx];
        
        // Fused Arithmetic: ((2*x + b) * x) + x
        float res = (2.0f * x + b) * x + x;
        output[idx] = res;
    }
}

// Convolution Transpose 3D kernel
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
    int total_output_elements = batch_size * out_channels * output_depth * output_height * output_width;

    if (out_idx >= total_output_elements) return;

    int tmp = out_idx;
    int out_w = tmp % output_width; tmp /= output_width;
    int out_h = tmp % output_height; tmp /= output_height;
    int out_d = tmp % output_depth; tmp /= output_depth;
    int out_c = tmp % out_channels; tmp /= out_channels;
    int batch = tmp;

    float sum = 0.0f;

    // Compute the range of kernel positions that contribute to this output position
    int k_size = kernel_size;
    int k_offset_start_d = max(0, (out_d + padding - (k_size - 1) + stride - 1) / stride);
    int k_offset_end_d = min(k_size, (out_d + padding + stride) / stride);
    int k_offset_start_h = max(0, (out_h + padding - (k_size - 1) + stride - 1) / stride);
    int k_offset_end_h = min(k_size, (out_h + padding + stride) / stride);
    int k_offset_start_w = max(0, (out_w + padding - (k_size - 1) + stride - 1) / stride);
    int k_offset_end_w = min(k_size, (out_w + padding + stride) / stride);

    for (int k_d = k_offset_start_d; k_d < k_offset_end_d; ++k_d) {
        for (int k_h = k_offset_start_h; k_h < k_offset_end_h; ++k_h) {
            for (int k_w = k_offset_start_w; k_w < k_offset_end_w; ++k_w) {
                int in_d = (out_d + padding - k_d) / stride;
                int in_h = (out_h + padding - k_h) / stride;
                int in_w = (out_w + padding - k_w) / stride;

                if ((out_d + padding - k_d) % stride == 0 &&
                    (out_h + padding - k_h) % stride == 0 &&
                    (out_w + padding - k_w) % stride == 0 &&
                    in_d >= 0 && in_d < input_depth &&
                    in_h >= 0 && in_h < input_height &&
                    in_w >= 0 && in_w < input_width) {
                    
                    int in_idx = batch * (in_channels * input_depth * input_height * input_width) +
                                 out_c * (input_depth * input_height * input_width) +
                                 in_d * (input_height * input_width) +
                                 in_h * input_width +
                                 in_w;
                    
                    int weight_idx = out_c * (in_channels * k_size * k_size * k_size) +
                                     out_c % in_channels * (k_size * k_size * k_size) +
                                     k_d * (k_size * k_size) +
                                     k_h * k_size +
                                     k_w;
                    
                    sum += input[in_idx] * weight[weight_idx];
                }
            }
        }
    }

    output[out_idx] = sum + bias[out_c];
}

void fused_op_forward(
    const torch::Tensor& input,
    const torch::Tensor& bias,
    torch::Tensor& output)
{
    const int num_elements = static_cast<int>(input.numel());
    const int spatial_size = static_cast<int>(input.size(2) * input.size(3) * input.size(4));
    const int out_channels = static_cast<int>(input.size(1));

    // Heuristic for grid size to keep SMs busy
    const int threads_per_block = 256;
    const int blocks = std::min((num_elements + threads_per_block - 1) / threads_per_block, 1024);

    fused_post_conv_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        num_elements,
        spatial_size,
        out_channels);
}

void conv_transpose3d_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    const int kernel_size,
    const int stride,
    const int padding,
    const int output_padding)
{
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int input_depth = input.size(2);
    const int input_height = input.size(3);
    const int input_width = input.size(4);
    
    const int out_channels = weight.size(1);
    const int output_depth = (input_depth - 1) * stride - 2 * padding + kernel_size + output_padding;
    const int output_height = (input_height - 1) * stride - 2 * padding + kernel_size + output_padding;
    const int output_width = (input_width - 1) * stride - 2 * padding + kernel_size + output_padding;

    const int threads_per_block = 256;
    const int total_output_elements = batch_size * out_channels * output_depth * output_height * output_width;
    const int blocks = std::min((total_output_elements + threads_per_block - 1) / threads_per_block, 1024);

    conv_transpose3d_kernel<<<blocks, threads_per_block>>>(
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
void fused_op_forward(const torch::Tensor& input, const torch::Tensor& bias, torch::Tensor& output);
void conv_transpose3d_forward(const torch::Tensor& input, const torch::Tensor& weight, const torch::Tensor& bias, 
                              torch::Tensor& output, const int kernel_size, const int stride, const int padding, const int output_padding);

torch::Tensor fused_op(const torch::Tensor& input, const torch::Tensor& bias) {
    auto output = torch::empty_like(input);
    fused_op_forward(input, bias, output);
    return output;
}

torch::Tensor conv_transpose3d_custom(const torch::Tensor& input, const torch::Tensor& weight, const torch::Tensor& bias,
                                       const int kernel_size, const int stride, const int padding, const int output_padding) {
    const int out_channels = weight.size(1);
    const int output_depth = (input.size(2) - 1) * stride - 2 * padding + kernel_size + output_padding;
    const int output_height = (input.size(3) - 1) * stride - 2 * padding + kernel_size + output_padding;
    const int output_width = (input.size(4) - 1) * stride - 2 * padding + kernel_size + output_padding;
    
    auto output = torch::empty({input.size(0), out_channels, output_depth, output_height, output_width}, 
                               torch::dtype(input.dtype()).device(input.device()));
    conv_transpose3d_forward(input, weight, bias, output, kernel_size, stride, padding, output_padding);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op, "Fused post-conv arithmetic with grid-stride loops");
    m.def("conv_transpose3d_custom", &conv_transpose3d_custom, "Custom Conv Transpose 3D");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_op_ext',
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
    # Execute transposed convolution via custom CUDA kernel
    x = fused_ext.conv_transpose3d_custom(
        x, 
        conv_transpose_weight, 
        conv_transpose_bias,
        kernel_size=3,
        stride=conv_transpose_stride[0],
        padding=conv_transpose_padding[0],
        output_padding=conv_transpose_output_padding[0]
    )

    return fused_ext.fused_op(x, bias.view(-1))
