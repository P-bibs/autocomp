# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_163027/code_1.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'output_padding', 'bias_shape', 'scaling_factor']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'bias', 'scaling_factor']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a transposed convolution, adds a bias term, clamps, scales, clamps, and divides.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor

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
    if 'scaling_factor' in flat_state:
        state_kwargs['scaling_factor'] = flat_state['scaling_factor']
    else:
        state_kwargs['scaling_factor'] = getattr(model, 'scaling_factor')
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

# Optimization: Fused kernel that performs Transposed Convolution + Bias + Clamps + Scaling in one pass
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_transpose_conv_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ conv_transpose_bias,
    const float* __restrict__ bias,
    float* __restrict__ output,
    float scaling_factor,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_height,
    int in_width,
    int kernel_size,
    int stride,
    int padding,
    int output_padding
) {
    int batch_idx = blockIdx.x;
    int out_channel = blockIdx.y;
    int out_pos = blockIdx.z * blockDim.x + threadIdx.x;

    if (batch_idx >= batch_size || out_channel >= out_channels || out_pos >= (in_height * stride) * (in_width * stride)) {
        return;
    }

    int out_height = in_height * stride;
    int out_width = in_width * stride;

    int out_y = out_pos / out_width;
    int out_x = out_pos % out_width;

    float acc = conv_transpose_bias[out_channel] + bias[out_channel];

    for (int in_c = 0; in_c < in_channels; ++in_c) {
        for (int ky = 0; ky < kernel_size; ++ky) {
            for (int kx = 0; kx < kernel_size; ++kx) {
                int in_y = out_y + padding - ky;
                int in_x = out_x + padding - kx;

                if (in_y % stride == 0 && in_x % stride == 0) {
                    in_y /= stride;
                    in_x /= stride;

                    if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
                        float val = input[batch_idx * (in_channels * in_height * in_width) +
                                          in_c * (in_height * in_width) +
                                          in_y * in_width + in_x];

                        float w = weight[out_channel * (in_channels * kernel_size * kernel_size) +
                                         in_c * (kernel_size * kernel_size) +
                                         (kernel_size - 1 - ky) * kernel_size +
                                         (kernel_size - 1 - kx)];

                        acc += val * w;
                    }
                }
            }
        }
    }

    // Apply clamping, scaling, and final clamping as in the original model
    acc = fmaxf(0.0f, fminf(1.0f, acc));
    acc *= scaling_factor;
    acc = fmaxf(0.0f, fminf(1.0f, acc));
    acc /= scaling_factor;

    output[batch_idx * (out_channels * out_height * out_width) +
           out_channel * (out_height * out_width) +
           out_y * out_width + out_x] = acc;
}

void fused_op_forward_kernel_launcher(
    const float* input,
    const float* weight,
    const float* conv_transpose_bias,
    const float* bias,
    float* output,
    float scaling_factor,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_height,
    int in_width,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    int threads
) {
    int out_height = in_height * stride;
    int out_width = in_width * stride;
    dim3 grid(batch_size, out_channels, (out_height * out_width + threads - 1) / threads);
    dim3 block(threads);

    fused_transpose_conv_kernel<<<grid, block>>>(
        input, weight, conv_transpose_bias, bias, output, scaling_factor,
        batch_size, in_channels, out_channels, in_height, in_width,
        kernel_size, stride, padding, output_padding
    );
}
"""

# --- C++ Logic (Interface/Bindings) ---
cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward_kernel_launcher(
    const float* input,
    const float* weight,
    const float* conv_transpose_bias,
    const float* bias,
    float* output,
    float scaling_factor,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_height,
    int in_width,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    int threads
);

void fused_op_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor conv_transpose_bias,
    torch::Tensor bias,
    torch::Tensor output,
    float scaling_factor,
    int stride,
    int padding,
    int output_padding,
    int threads
) {
    fused_op_forward_kernel_launcher(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        conv_transpose_bias.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        scaling_factor,
        input.size(0),
        input.size(1),
        weight.size(1),
        input.size(2),
        input.size(3),
        weight.size(2),
        stride,
        padding,
        output_padding,
        threads
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused Transpose Conv + Bias + Clamp + Scale");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_op',
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
    scaling_factor,
):
    assert conv_transpose_groups == 1
    assert conv_transpose_dilation == 1
    
    batch_size, in_channels, in_height, in_width = x.shape
    out_channels = conv_transpose_weight.shape[1]
    out_height = (in_height - 1) * conv_transpose_stride + conv_transpose_weight.shape[2] - 2 * conv_transpose_padding + conv_transpose_output_padding
    out_width = (in_width - 1) * conv_transpose_stride + conv_transpose_weight.shape[3] - 2 * conv_transpose_padding + conv_transpose_output_padding

    output = torch.empty((batch_size, out_channels, out_height, out_width), device=x.device, dtype=x.dtype)

    # Flatten bias to match expected shape in kernel
    bias_flat = bias.flatten()

    fused_ext.fused_op(
        x,
        conv_transpose_weight,
        conv_transpose_bias,
        bias_flat,
        output,
        scaling_factor,
        conv_transpose_stride,
        conv_transpose_padding,
        conv_transpose_output_padding,
        256  # threads per block
    )

    return output

batch_size = 128
in_channels  = 64  
out_channels = 64  
height = width = 128 
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1)
scaling_factor = 2.0

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]
