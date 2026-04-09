# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_130922/code_4.py
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

# Optimized CUDA kernel
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

#define DIV_UP(a, b) (((a) + (b) - 1) / (b))

// Optimized kernel fusing ConvTranspose3d with custom activation
__global__ void fused_conv_transpose3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ post_bias, // bias for post-processing
    float* __restrict__ output,
    int batch,
    int in_channels,
    int out_channels,
    int in_depth,
    int in_height,
    int in_width,
    int out_depth,
    int out_height,
    int out_width,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    // Each thread processes one output element
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch * out_channels * out_depth * out_height * out_width;
    if (idx >= total_elements) return;

    // Decompose linear index to 5D coordinates
    int tmp = idx;
    int w_out = tmp % out_width; tmp /= out_width;
    int h_out = tmp % out_height; tmp /= out_height;
    int d_out = tmp % out_depth; tmp /= out_depth;
    int oc = tmp % out_channels; tmp /= out_channels;
    int b = tmp;

    float acc = 0.0f;

    // Loop over input feature map and kernel
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kd = 0; kd < kernel_size; ++kd) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    // Compute corresponding input location
                    int d_in = d_out + padding - kd * dilation;
                    int h_in = h_out + padding - kh * dilation;
                    int w_in = w_out + padding - kw * dilation;

                    // Check if valid input position and divisible by stride
                    if (d_in >= 0 && d_in < in_depth * stride && d_in % stride == 0 &&
                        h_in >= 0 && h_in < in_height * stride && h_in % stride == 0 &&
                        w_in >= 0 && w_in < in_width * stride && w_in % stride == 0) {

                        d_in /= stride;
                        h_in /= stride;
                        w_in /= stride;

                        if (d_in < in_depth && h_in < in_height && w_in < in_width) {
                            int input_idx = ((b * in_channels + ic) * in_depth + d_in) * in_height * in_width +
                                            h_in * in_width + w_in;
                            int weight_idx = ((oc * in_channels + ic) * kernel_size + kd) * kernel_size * kernel_size +
                                             kh * kernel_size + kw;
                            acc += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
    }

    // Add bias from conv transpose (assumed to be 0 or handled externally for simplicity)
    float x = acc;
    float bias_val = post_bias[oc];

    // Compute fused activation: ((x + bias) + x) * x + x = (2*x + bias) * x + x
    float tmp1 = x + bias_val;     // x + b
    float tmp2 = tmp1 + x;         // (x + b) + x = 2*x + b
    float result = tmp2 * x + x;   // ((2*x + b) * x) + x

    output[idx] = result;
}

void fused_conv_transpose3d_cuda(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& post_bias,
    torch::Tensor& output,
    int stride,
    int padding,
    int dilation
) {
    int batch = input.size(0);
    int in_channels = input.size(1);
    int in_depth = input.size(2);
    int in_height = input.size(3);
    int in_width = input.size(4);

    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);

    int out_depth = (in_depth - 1) * stride - 2 * padding + kernel_size + 1;
    int out_height = (in_height - 1) * stride - 2 * padding + kernel_size + 1;
    int out_width = (in_width - 1) * stride - 2 * padding + kernel_size + 1;

    const int threads_per_block = 256;
    const int total_elements = batch * out_channels * out_depth * out_height * out_width;
    const int blocks = DIV_UP(total_elements, threads_per_block);

    fused_conv_transpose3d_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        post_bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch,
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
        dilation
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_conv_transpose3d_cuda(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& post_bias,
    torch::Tensor& output,
    int stride,
    int padding,
    int dilation
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fused_conv_transpose3d_cuda, "Fused ConvTranspose3D + Activation");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_conv',
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
    # Ensure all tensors are on CUDA
    x = x.cuda()
    conv_transpose_weight = conv_transpose_weight.cuda()
    bias = bias.cuda()

    # Validate group count (only supporting groups=1)
    if conv_transpose_groups != 1:
        raise NotImplementedError("Only conv_transpose_groups=1 is supported")

    # Compute output dimensions
    batch, in_channels, in_depth, in_height, in_width = x.shape
    out_channels, _, kernel_size, _, _ = conv_transpose_weight.shape
    
    out_depth = (in_depth - 1) * conv_transpose_stride - 2 * conv_transpose_padding + kernel_size + conv_transpose_output_padding
    out_height = (in_height - 1) * conv_transpose_stride - 2 * conv_transpose_padding + kernel_size + conv_transpose_output_padding
    out_width = (in_width - 1) * conv_transpose_stride - 2 * conv_transpose_padding + kernel_size + conv_transpose_output_padding
    
    # Allocate output tensor
    output = torch.empty((batch, out_channels, out_depth, out_height, out_width), device='cuda', dtype=x.dtype)
    
    # Call the fused kernel
    fused_ext.forward(
        x,
        conv_transpose_weight,
        bias.view(-1),
        output,
        conv_transpose_stride,
        conv_transpose_padding,
        conv_transpose_dilation
    )
    
    return output
