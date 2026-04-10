# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_152515/code_2.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'add_value', 'multiply_value']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'add_value', 'multiply_value']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a transposed convolution, adds a value, takes the minimum, applies GELU, and multiplies by a value.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, add_value, multiply_value):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.add_value = add_value
        self.multiply_value = multiply_value

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
    if 'add_value' in flat_state:
        state_kwargs['add_value'] = flat_state['add_value']
    else:
        state_kwargs['add_value'] = getattr(model, 'add_value')
    if 'multiply_value' in flat_state:
        state_kwargs['multiply_value'] = flat_state['multiply_value']
    else:
        state_kwargs['multiply_value'] = getattr(model, 'multiply_value')
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
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

# --- CUDA Kernel for fused convolution transpose and element-wise operations ---
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

#define GELU_SCALING_FACTOR 0.7978845608028654f // sqrt(2.0f / M_PI)

// Fast GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
__device__ __forceinline__ float fast_gelu(float x) {
    const float sqrt_2_over_pi = GELU_SCALING_FACTOR;
    const float kappa = 0.044715f;
    float x_cubed = x * x * x;
    float inner = sqrt_2_over_pi * (x + kappa * x_cubed);
    float tanh_inner = tanhf(inner);
    return 0.5f * x * (1.0f + tanh_inner);
}

// Conv2d_transpose forward pass logic adapted for our kernel
__global__ void fused_conv_transpose2d_elementwise_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width,
    const int kernel_size,
    const int stride,
    const int padding,
    const int output_padding,
    const int groups,
    const int dilation,
    const float add_value,
    const float multiply_value
) {
    const int out_ch_per_group = out_channels / groups;
    const int in_ch_per_group = in_channels / groups;

    const int out_pixel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_out_pixels = batch_size * out_channels * output_height * output_width;

    if (out_pixel_idx >= total_out_pixels) return;

    const int w_out = out_pixel_idx % output_width;
    const int h_out = (out_pixel_idx / output_width) % output_height;
    const int c_out = (out_pixel_idx / (output_width * output_height)) % out_channels;
    const int n = out_pixel_idx / (output_width * output_height * out_channels);

    const int group_id = c_out / out_ch_per_group;
    const int g_c_out = c_out % out_ch_per_group;

    float conv_sum = (bias != nullptr) ? bias[c_out] : 0.0f;

    // Compute input coordinates for convolution transpose
    const int h_in_start = (h_out + padding - (kernel_size - 1) * dilation + stride - 1) / stride;
    const int h_in_end = (h_out + padding) / stride;
    const int w_in_start = (w_out + padding - (kernel_size - 1) * dilation + stride - 1) / stride;
    const int w_in_end = (w_out + padding) / stride;

    for (int h_in = h_in_start; h_in <= h_in_end; ++h_in) {
        for (int w_in = w_in_start; w_in <= w_in_end; ++w_in) {
            if (h_in < 0 || h_in >= input_height || w_in < 0 || w_in >= input_width) continue;

            const int k_h = h_out + padding - h_in * stride;
            const int k_w = w_out + padding - w_in * stride;

            if (k_h % dilation != 0 || k_w % dilation != 0) continue;
            const int k_h_dilated = k_h / dilation;
            const int k_w_dilated = k_w / dilation;

            if (k_h_dilated < 0 || k_h_dilated >= kernel_size || k_w_dilated < 0 || k_w_dilated >= kernel_size) continue;

            const int w_idx = ((group_id * out_ch_per_group + g_c_out) * in_ch_per_group + 0) * kernel_size * kernel_size +
                              k_h_dilated * kernel_size + k_w_dilated;
            const int i_idx = n * in_channels * input_height * input_width +
                              (group_id * in_ch_per_group + 0) * input_height * input_width +
                              h_in * input_width + w_in;

            for (int c_in_grp = 0; c_in_grp < in_ch_per_group; ++c_in_grp) {
                int w_idx_corr = ((group_id * out_ch_per_group + g_c_out) * in_ch_per_group + c_in_grp) * kernel_size * kernel_size +
                                 k_h_dilated * kernel_size + k_w_dilated;
                int i_idx_corr = n * in_channels * input_height * input_width +
                                 (group_id * in_ch_per_group + c_in_grp) * input_height * input_width +
                                 h_in * input_width + w_in;
                conv_sum += weight[w_idx_corr] * input[i_idx_corr];
            }
        }
    }

    // Fused element-wise operations
    float val = conv_sum + add_value;
    val = fminf(val, 0.0f); // min(x, 0.0)
    val = fast_gelu(val);    // gelu(x)
    val = val * multiply_value;

    output[out_pixel_idx] = val;
}

void fused_conv_transpose2d_elementwise_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    int groups,
    int dilation,
    float add_value,
    float multiply_value
) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int input_height = input.size(2);
    const int input_width = input.size(3);
    const int out_channels = weight.size(1); // Note: Transposed conv weight shape is [in_ch, out_ch/groups, kH, kW]
    const int output_height = (input_height - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1;
    const int output_width = (input_width - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1;

    const int total_out_pixels = batch_size * out_channels * output_height * output_width;
    const int threads = 256;
    const int blocks = (total_out_pixels + threads - 1) / threads;

    fused_conv_transpose2d_elementwise_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_height,
        input_width,
        output_height,
        output_width,
        kernel_size,
        stride,
        padding,
        output_padding,
        groups,
        dilation,
        add_value,
        multiply_value
    );
}
"""

# --- C++ Logic (Interface/Bindings) ---
cpp_source = r"""
#include <torch/extension.h>

void fused_conv_transpose2d_elementwise_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    int groups,
    int dilation,
    float add_value,
    float multiply_value
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_transpose2d_elementwise", &fused_conv_transpose2d_elementwise_forward, "Fused Conv Transpose 2D and Element-wise Ops");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_conv_transpose2d_elementwise_op',
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
    add_value,
    multiply_value,
):
    # Calculate output dimensions for conv_transpose2d
    kernel_size = conv_transpose_weight.shape[2] # Assuming square kernel
    output_height = (x.shape[2] - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_dilation * (kernel_size - 1) + conv_transpose_output_padding + 1
    output_width = (x.shape[3] - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_dilation * (kernel_size - 1) + conv_transpose_output_padding + 1
    out_channels = conv_transpose_weight.shape[1]
    
    # Allocate output tensor
    output = torch.empty((x.shape[0], out_channels, output_height, output_width), device=x.device, dtype=x.dtype)
    
    # Call the fused kernel
    fused_ext.fused_conv_transpose2d_elementwise(
        x, conv_transpose_weight, conv_transpose_bias, output,
        kernel_size, conv_transpose_stride, conv_transpose_padding,
        conv_transpose_output_padding, conv_transpose_groups,
        conv_transpose_dilation, add_value, multiply_value
    )
    
    return output

# Parameters (unchanged)
batch_size = 128
in_channels = 64
out_channels = 128
height, width = 64, 64
kernel_size = 4
stride = 2
add_value = 0.5
multiply_value = 2.0

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, add_value, multiply_value]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

