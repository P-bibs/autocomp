# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_161645/code_1.py
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

# Optimization: Kernel Fusion (Optimization #8)
# We fuse bias addition, double clamping, and scaling into the convolution loop.

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_conv_transpose_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias_tensor,
    float* __restrict__ output,
    float scaling_factor,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_size,
    int stride,
    int padding,
    int output_padding
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * out_height * out_width) return;

    // Compute output coordinates from flat index
    int temp = idx;
    int ow = temp % out_width; temp /= out_width;
    int oh = temp % out_height; temp /= out_height;
    int oc = temp % out_channels;
    int b  = temp / out_channels;

    // Initialize accumulator with bias
    float val = bias_tensor[oc];

    // Convolution loop - transpose logic
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                // Map output position to input position based on transposed convolution formula
                // https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html
                int ih = oh - (kh - padding) * 1; // dilation=1
                int iw = ow - (kw - padding) * 1;

                // Check if input coordinates are valid and divisible by stride
                if (ih % stride == 0 && iw % stride == 0) {
                    ih /= stride;
                    iw /= stride;

                    if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                        // Input tensor access: [batch][channel][height][width]
                        int input_idx = ((b * in_channels + ic) * in_height + ih) * in_width + iw;
                        // Weight tensor access: [out_channel][in_channel/group][kh][kw]
                        // Note: This assumes groups=1
                        int weight_idx = ((ic * out_channels + oc) * kernel_size + kh) * kernel_size + kw;
                        val += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }

    // Apply fused element-wise operations:
    // 1. First clamp to [0, 1]
    val = fminf(fmaxf(val, 0.0f), 1.0f);
    
    // 2. Scale
    val *= scaling_factor;
    
    // 3. Second clamp to [0, 1]
    val = fminf(fmaxf(val, 0.0f), 1.0f);
    
    // 4. Divide by scaling factor
    val /= scaling_factor;

    // Write result
    output[idx] = val;
}

void fused_op_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias_tensor,
    torch::Tensor output,
    float scaling_factor,
    int kernel_size,
    int stride,
    int padding,
    int output_padding
) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int in_height = input.size(2);
    const int in_width = input.size(3);
    
    const int out_channels = weight.size(1); // For ConvTranspose2d: weight is [in_channels, out_channels/groups, kH, kW]
    const int out_height = output.size(2);
    const int out_width = output.size(3);

    const int total_elements = batch_size * out_channels * out_height * out_width;
    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;

    fused_conv_transpose_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_tensor.data_ptr<float>(),
        output.data_ptr<float>(),
        scaling_factor,
        batch_size,
        in_channels,
        out_channels,
        in_height,
        in_width,
        out_height,
        out_width,
        kernel_size,
        stride,
        padding,
        output_padding
    );
}
"""

# --- C++ Logic (Interface/Bindings) ---
cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias_tensor,
    torch::Tensor output,
    float scaling_factor,
    int kernel_size,
    int stride,
    int padding,
    int output_padding
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused Conv Transpose Forward");
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

# ConvTranspose2D weight shape: [in_channels, out_channels/groups, kernel_size, kernel_size]
weight_shape = (in_channels, out_channels, kernel_size, kernel_size)
bias_tensor_shape = (out_channels,)

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
    assert conv_transpose_groups == 1, "Groups != 1 not supported in this implementation"
    assert conv_transpose_dilation == 1, "Dilation != 1 not supported in this implementation"
    
    device = x.device
    batch_size = x.shape[0]
    
    # Calculate output size manually based on ConvTranspose2d formula
    out_h = (x.shape[2] - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_weight.shape[2] + conv_transpose_output_padding
    out_w = (x.shape[3] - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_weight.shape[3] + conv_transpose_output_padding
    output_shape = (batch_size, conv_transpose_weight.shape[1], out_h, out_w)

    out = torch.empty(output_shape, device=device)
    fused_ext.fused_op(
        x.contiguous(),
        conv_transpose_weight.contiguous(),
        bias.view(-1).contiguous(),  # Flatten bias to [out_channels]
        out,
        scaling_factor,
        conv_transpose_weight.shape[2],  # Kernel size (assuming square)
        conv_transpose_stride,
        conv_transpose_padding,
        conv_transpose_output_padding
    )
    return out
