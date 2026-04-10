# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_163518/code_1.py
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

# CUDA kernel for fused transpose convolution with post-processing
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

__global__ void fused_conv_transpose_clamp_scale_clamp_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ bias,
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
    int padding
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (out_idx >= batch_size * out_channels * out_height * out_width) return;
    
    int tmp = out_idx;
    const int w_out = tmp % out_width; tmp /= out_width;
    const int h_out = tmp % out_height; tmp /= out_height;
    const int c_out = tmp % out_channels;
    const int n = tmp / out_channels;
    
    const int h_in_start = (h_out + padding - kernel_size + 1 > 0) ? (h_out + padding - kernel_size + 1 + stride - 1) / stride : 0;
    const int h_in_end = min((h_out + padding) / stride + 1, in_height);
    const int w_in_start = (w_out + padding - kernel_size + 1 > 0) ? (w_out + padding - kernel_size + 1 + stride - 1) / stride : 0;
    const int w_in_end = min((w_out + padding) / stride + 1, in_width);
    
    float val = 0.0f;
    
    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int h_in = h_in_start; h_in < h_in_end; ++h_in) {
            for (int w_in = w_in_start; w_in < w_in_end; ++w_in) {
                const int kh = h_out - h_in * stride + padding;
                const int kw = w_out - w_in * stride + padding;
                
                if (kh < 0 || kh >= kernel_size || kw < 0 || kw >= kernel_size) continue;
                
                const int input_idx = ((n * in_channels + c_in) * in_height + h_in) * in_width + w_in;
                const int weight_idx = ((c_out * in_channels + c_in) * kernel_size + kh) * kernel_size + kw;
                
                val += input[input_idx] * weight[weight_idx];
            }
        }
    }
    
    // Add convolution bias
    val += conv_bias[c_out];
    
    // Add additional bias
    val += bias[c_out];
    
    // First clamp: clamp to [0, 1]
    val = fmaxf(0.0f, fminf(1.0f, val));
    
    // Scale by factor
    val *= scaling_factor;
    
    // Second clamp: clamp to [0, 1]
    val = fmaxf(0.0f, fminf(1.0f, val));
    
    // Divide by factor
    val /= scaling_factor;
    
    output[out_idx] = val;
}

void launch_fused_conv_transpose_clamp_scale_clamp(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor conv_bias,
    torch::Tensor bias,
    torch::Tensor output,
    float scaling_factor,
    int kernel_size,
    int stride,
    int padding
) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int in_height = input.size(2);
    const int in_width = input.size(3);
    
    const int out_channels = weight.size(1);
    const int out_height = output.size(2);
    const int out_width = output.size(3);
    
    const int total_threads = batch_size * out_channels * out_height * out_width;
    const int threads_per_block = 256;
    const int num_blocks = (total_threads + threads_per_block - 1) / threads_per_block;
    
    fused_conv_transpose_clamp_scale_clamp_kernel<<<num_blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        conv_bias.data_ptr<float>(),
        bias.data_ptr<float>(),
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
        padding
    );
    
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""

# C++ wrapper
cpp_source = r"""
#include <torch/extension.h>

void launch_fused_conv_transpose_clamp_scale_clamp(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor conv_bias,
    torch::Tensor bias,
    torch::Tensor output,
    float scaling_factor,
    int kernel_size,
    int stride,
    int padding
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_transpose_clamp_scale_clamp", 
          &launch_fused_conv_transpose_clamp_scale_clamp, 
          "Fused ConvTranspose2d with clamp-scale-clamp operations");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_conv_transpose_ops',
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
    # Validate that we're using groups=1 and dilation=1 (simplifying assumptions for this implementation)
    assert conv_transpose_groups == 1, "Only groups=1 is supported"
    assert conv_transpose_dilation == 1, "Only dilation=1 is supported"
    
    # Compute output dimensions
    batch_size = x.size(0)
    out_channels = conv_transpose_weight.size(1)
    kernel_size = conv_transpose_weight.size(2)
    stride = conv_transpose_stride
    padding = conv_transpose_padding
    output_padding = conv_transpose_output_padding
    
    out_height = (x.size(2) - 1) * stride - 2 * padding + kernel_size + output_padding
    out_width = (x.size(3) - 1) * stride - 2 * padding + kernel_size + output_padding
    
    # Create output tensor
    output = torch.empty((batch_size, out_channels, out_height, out_width), device=x.device, dtype=x.dtype)
    
    # Launch fused kernel
    fused_ext.fused_conv_transpose_clamp_scale_clamp(
        x,
        conv_transpose_weight,
        conv_transpose_bias,
        bias,
        output,
        scaling_factor,
        kernel_size,
        stride,
        padding
    )
    
    return output

# Constants (unchanged from original)
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
