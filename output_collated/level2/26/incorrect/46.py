# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_045050/code_7.py
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
from torch.utils.cpp_extension import load_inline

# ----------------------------------------------------------------------
# CUDA kernel: Fused Conv3D Transpose + Add + HardSwish
# ----------------------------------------------------------------------

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

#define CUDA_1D_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

__global__ void fused_conv_transpose3d_add_hardswish_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ add_input,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int input_d, const int input_h, const int input_w,
    const int output_d, const int output_h, const int output_w,
    const int kernel_size,
    const int stride,
    const int padding,
    const int output_padding
) {
    // Each thread computes one output element
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = batch_size * out_channels * output_d * output_h * output_w;

    if (out_idx >= total_outputs) return;

    // Decompose output index
    int tmp = out_idx;
    const int w_out = tmp % output_w; tmp /= output_w;
    const int h_out = tmp % output_h; tmp /= output_h;
    const int d_out = tmp % output_d; tmp /= output_d;
    const int c_out = tmp % out_channels; tmp /= out_channels;
    const int b = tmp;

    // Compute input coordinates for this output position
    const int d_in_start = (d_out + padding - (kernel_size - 1) + stride - 1) / stride;
    const int h_in_start = (h_out + padding - (kernel_size - 1) + stride - 1) / stride;
    const int w_in_start = (w_out + padding - (kernel_size - 1) + stride - 1) / stride;

    const int d_in_end = min((d_out + padding) / stride + 1, input_d);
    const int h_in_end = min((h_out + padding) / stride + 1, input_h);
    const int w_in_end = min((w_out + padding) / stride + 1, input_w);

    float acc = 0.0f;

    // Loop over input points that contribute to this output
    for (int d_in = d_in_start; d_in < d_in_end; d_in++) {
        for (int h_in = h_in_start; h_in < h_in_end; h_in++) {
            for (int w_in = w_in_start; w_in < w_in_end; w_in++) {
                // Calculate kernel indices
                const int kd = d_out - d_in * stride + padding;
                const int kh = h_out - h_in * stride + padding;
                const int kw = w_out - w_in * stride + padding;

                // Check bounds for kernel
                if (kd >= 0 && kd < kernel_size &&
                    kh >= 0 && kh < kernel_size &&
                    kw >= 0 && kw < kernel_size) {

                    // Weight index: [out_ch, in_ch, kD, kH, kW]
                    const int weight_idx = 
                        c_out * (in_channels * kernel_size * kernel_size * kernel_size) +
                        0 * (kernel_size * kernel_size * kernel_size) +
                        kd * (kernel_size * kernel_size) +
                        kh * kernel_size +
                        kw;

                    // Input index: [batch, in_ch, d, h, w]
                    const int input_idx = 
                        b * (in_channels * input_d * input_h * input_w) +
                        0 * (input_d * input_h * input_w) +
                        d_in * (input_h * input_w) +
                        h_in * input_w +
                        w_in;

                    // Loop over input channels
                    for (int c_in = 0; c_in < in_channels; c_in++) {
                        const int w_idx = weight_idx + c_in * (kernel_size * kernel_size * kernel_size);
                        const int i_idx = input_idx + c_in * (input_d * input_h * input_w);
                        acc += input[i_idx] * weight[w_idx];
                    }
                }
            }
        }
    }

    // Add bias
    acc += bias[c_out];

    // Add from add_input tensor
    const int add_idx = b * (out_channels * output_d * output_h * output_w) +
                        c_out * (output_d * output_h * output_w) +
                        d_out * (output_h * output_w) +
                        h_out * output_w +
                        w_out;
    
    float val = acc + add_input[add_idx];

    // Apply HardSwish: x * clamp(x + 3, 0, 6) / 6
    float relu6_val = fminf(fmaxf(val + 3.0f, 0.0f), 6.0f);
    float hardswish = val * relu6_val * 0.16666666666666666f;

    output[out_idx] = hardswish;
}

void fused_conv_transpose3d_add_hardswish_launcher(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor add_input,
    torch::Tensor output,
    int kernel_size,
    int stride,
    int padding,
    int output_padding
) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int input_d = input.size(2);
    const int input_h = input.size(3);
    const int input_w = input.size(4);

    const int out_channels = weight.size(0);
    const int output_d = add_input.size(2);
    const int output_h = add_input.size(3);
    const int output_w = add_input.size(4);

    const int total_outputs = batch_size * out_channels * output_d * output_h * output_w;
    const int threads = 256;
    const int blocks = (total_outputs + threads - 1) / threads;

    fused_conv_transpose3d_add_hardswish_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        add_input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_d, input_h, input_w,
        output_d, output_h, output_w,
        kernel_size,
        stride,
        padding,
        output_padding
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_conv_transpose3d_add_hardswish_launcher(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor add_input,
    torch::Tensor output,
    int kernel_size,
    int stride,
    int padding,
    int output_padding
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_transpose3d_add_hardswish", 
          &fused_conv_transpose3d_add_hardswish_launcher, 
          "Fused Conv3D Transpose + Add + HardSwish");
}
"""

fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# Parameters matching the original
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
    return [
        torch.rand(batch_size, in_channels, D, H, W, device='cuda'), 
        torch.rand(batch_size, out_channels, D*stride, H*stride, W*stride, device='cuda')
    ]

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
    # Ensure all inputs are contiguous
    x = x.contiguous()
    add_input = add_input.contiguous()
    conv_transpose_weight = conv_transpose_weight.contiguous()
    conv_transpose_bias = conv_transpose_bias.contiguous()
    
    # Create output tensor
    out = torch.empty_like(add_input)
    
    # Call the fused kernel
    fused_ext.fused_conv_transpose3d_add_hardswish(
        x,
        conv_transpose_weight,
        conv_transpose_bias,
        add_input,
        out,
        kernel_size,
        conv_transpose_stride[0],  # Assuming symmetric
        conv_transpose_padding[0],  # Assuming symmetric
        conv_transpose_output_padding[0]  # Assuming symmetric
    )
    
    return out
