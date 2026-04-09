# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_112403/code_2.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'bias_shape', 'stride', 'padding', 'output_padding']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'bias']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a transposed convolution, subtracts a bias term, and applies tanh activation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, bias_shape, stride=2, padding=1, output_padding=1):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
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
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# --- CUDA Kernel Code ---
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

template <typename scalar_t>
__device__ __forceinline__ scalar_t fast_tanh(scalar_t x) {
    // Fast tanh approximation using tanh_sinh or other methods if needed
    // For now using standard tanh, but compiled with --use_fast_math
    return tanh(x);
}

template <typename scalar_t>
__global__ void fused_conv_transpose2d_tanh_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ conv_bias,
    const scalar_t* __restrict__ subtract_bias,
    scalar_t* __restrict__ output,
    int batch_size,
    int in_channels,
    int in_height,
    int in_width,
    int out_channels,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int output_padding_h,
    int output_padding_w,
    int dilation_h,
    int dilation_w,
    int groups,
    int out_height,
    int out_width
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * out_channels * out_height * out_width;

    if (tid >= total_elements) return;

    const int n = tid / (out_channels * out_height * out_width);
    const int c = (tid / (out_height * out_width)) % out_channels;
    const int h = (tid / out_width) % out_height;
    const int w = tid % out_width;

    scalar_t sum = 0.0;

    const int group_id = c / (out_channels / groups);
    const int weight_offset_per_group = (in_channels / groups) * out_channels / groups * kernel_h * kernel_w;

    for (int ic = 0; ic < in_channels / groups; ++ic) {
        const int input_channel = group_id * (in_channels / groups) + ic;
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                const int ih = h * stride_h - padding_h + kh * dilation_h;
                const int iw = w * stride_w - padding_w + kw * dilation_w;

                if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                    const int input_idx = n * in_channels * in_height * in_width +
                                          input_channel * in_height * in_width +
                                          ih * in_width + iw;
                    const int weight_idx = group_id * weight_offset_per_group +
                                           ic * (out_channels / groups) * kernel_h * kernel_w +
                                           (c % (out_channels / groups)) * kernel_h * kernel_w +
                                           kh * kernel_w + kw;

                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    // Add convolution bias
    sum += conv_bias[c];

    // Subtract our custom bias
    sum -= subtract_bias[c];

    // Apply tanh activation
    output[tid] = fast_tanh(sum);
}

void launch_fused_op(
    const torch::Tensor &input,
    const torch::Tensor &weight,
    const torch::Tensor &conv_bias,
    const torch::Tensor &subtract_bias,
    torch::Tensor &output,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int output_padding_h, int output_padding_w,
    int dilation_h, int dilation_w,
    int groups
) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int in_height = input.size(2);
    const int in_width = input.size(3);

    const int out_channels = weight.size(1); // Note: transpose conv weight shape [in_ch, out_ch/group, kH, kW]
    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3);

    const int out_height = (in_height - 1) * stride_h - 2 * padding_h + dilation_h * (kernel_h - 1) + output_padding_h + 1;
    const int out_width = (in_width - 1) * stride_w - 2 * padding_w + dilation_w * (kernel_w - 1) + output_padding_w + 1;

    const int total_threads = batch_size * out_channels * out_height * out_width;
    const int threads_per_block = 512;
    const int blocks = (total_threads + threads_per_block - 1) / threads_per_block;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "fused_conv_transpose2d_tanh_kernel", ([&] {
        fused_conv_transpose2d_tanh_kernel<scalar_t><<<blocks, threads_per_block>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            conv_bias.data_ptr<scalar_t>(),
            subtract_bias.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            in_channels,
            in_height,
            in_width,
            out_channels,
            kernel_h,
            kernel_w,
            stride_h, stride_w,
            padding_h, padding_w,
            output_padding_h, output_padding_w,
            dilation_h, dilation_w,
            groups,
            out_height,
            out_width
        );
    }));
}
"""

# --- C++ Logic (Interface/Bindings) ---
cpp_source = r"""
#include <torch/extension.h>

void launch_fused_op(
    const torch::Tensor &input,
    const torch::Tensor &weight,
    const torch::Tensor &conv_bias,
    const torch::Tensor &subtract_bias,
    torch::Tensor &output,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int output_padding_h, int output_padding_w,
    int dilation_h, int dilation_w,
    int groups
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &launch_fused_op, "Fused ConvTranspose2d + Bias Sub + Tanh");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_conv_transpose_tanh',
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
    # Ensure all tensors are on the same device and have the same dtype
    device = x.device
    dtype = x.dtype
    
    # Handle stride, padding, output_padding, dilation tuples
    if isinstance(conv_transpose_stride, int):
        stride_h = stride_w = conv_transpose_stride
    else:
        stride_h, stride_w = conv_transpose_stride[0], conv_transpose_stride[1]
        
    if isinstance(conv_transpose_padding, int):
        padding_h = padding_w = conv_transpose_padding
    else:
        padding_h, padding_w = conv_transpose_padding[0], conv_transpose_padding[1]
        
    if isinstance(conv_transpose_output_padding, int):
        output_padding_h = output_padding_w = conv_transpose_output_padding
    else:
        output_padding_h, output_padding_w = conv_transpose_output_padding[0], conv_transpose_output_padding[1]
        
    if isinstance(conv_transpose_dilation, int):
        dilation_h = dilation_w = conv_transpose_dilation
    else:
        dilation_h, dilation_w = conv_transpose_dilation[0], conv_transpose_dilation[1]

    # Create output tensor with correct shape
    out_channels = conv_transpose_weight.size(1)
    in_height, in_width = x.size(2), x.size(3)
    out_height = (in_height - 1) * stride_h - 2 * padding_h + dilation_h * (conv_transpose_weight.size(2) - 1) + output_padding_h + 1
    out_width = (in_width - 1) * stride_w - 2 * padding_w + dilation_w * (conv_transpose_weight.size(3) - 1) + output_padding_w + 1
    
    output = torch.empty((x.size(0), out_channels, out_height, out_width), device=device, dtype=dtype)

    # Call the fused kernel
    fused_ext.fused_op(
        x,
        conv_transpose_weight,
        conv_transpose_bias,
        bias.expand_as(conv_transpose_bias), # Expand bias to match conv_bias shape
        output,
        stride_h, stride_w,
        padding_h, padding_w,
        output_padding_h, output_padding_w,
        dilation_h, dilation_w,
        conv_transpose_groups
    )
    
    return output

# Test parameters
batch_size = 32
in_channels  = 64  
out_channels = 64  
height = width = 256 
kernel_size = 4
bias_shape = (out_channels, 1, 1)

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]
