# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_163844/code_1.py
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

# CUDA kernel: Fused deconvolution + point-wise chain
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_deconv_add_clamp_scale_clamp_div_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ add_bias,
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
    // Output pixel index
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= batch_size * out_channels * out_height * out_width) return;

    int oc = (out_idx / (out_height * out_width)) % out_channels;
    int oh = (out_idx / out_width) % out_height;
    int ow = out_idx % out_width;
    int b = out_idx / (out_channels * out_height * out_width);

    float val = 0.0f;

    // Compute input range that contributes to this output pixel
    // For each position in the kernel
    for (int ky = 0; ky < kernel_size; ky++) {
        for (int kx = 0; kx < kernel_size; kx++) {
            // Compute corresponding input position
            int ih = oh - stride * ky + padding;
            int iw = ow - stride * kx + padding;
            
            // Check if it's in valid input bounds
            if (ih % stride == 0 && iw % stride == 0) {
                ih /= stride;
                iw /= stride;
                
                if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                    for (int ic = 0; ic < in_channels; ic++) {
                        int input_idx = b * (in_channels * in_height * in_width) +
                                        ic * (in_height * in_width) +
                                        ih * in_width + iw;
                        int weight_idx = oc * (in_channels * kernel_size * kernel_size) +
                                         ic * (kernel_size * kernel_size) +
                                         ky * kernel_size + kx;
                        
                        val += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }

    // Add convolution bias
    val += conv_bias[oc];

    // Apply post-processing chain: add bias, clamp, scale, clamp, divide
    val += add_bias[oc];
    val = fmaxf(0.0f, fminf(1.0f, val));  // clamp to [0, 1]
    val *= scaling_factor;
    val = fmaxf(0.0f, fminf(1.0f, val));  // clamp to [0, 1]
    val /= scaling_factor;

    output[out_idx] = val;
}

void fused_op_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor conv_bias,
    torch::Tensor add_bias,
    torch::Tensor output,
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
    const int total_threads = batch_size * out_channels * out_height * out_width;
    const int threads_per_block = 256;
    const int blocks = (total_threads + threads_per_block - 1) / threads_per_block;

    fused_deconv_add_clamp_scale_clamp_div_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        conv_bias.data_ptr<float>(),
        add_bias.data_ptr<float>(),
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
}
"""

# --- C++ Logic (Interface/Bindings) ---
cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor conv_bias,
    torch::Tensor add_bias,
    torch::Tensor output,
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
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused deconv + add + clamp + scale + clamp + div kernel");
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
    batch_size, in_channels, in_height, in_width = x.shape
    out_channels, _, kernel_size, _ = conv_transpose_weight.shape
    
    # Calculate output dimensions for transposed convolution with stride=2, padding=1, output_padding=1
    out_height = (in_height - 1) * conv_transpose_stride[0] - 2 * conv_transpose_padding[0] + kernel_size + conv_transpose_output_padding[0]
    out_width = (in_width - 1) * conv_transpose_stride[1] - 2 * conv_transpose_padding[1] + kernel_size + conv_transpose_output_padding[1]
    
    output = torch.empty((batch_size, out_channels, out_height, out_width), device=x.device, dtype=x.dtype)
    
    # Call the fused kernel
    fused_ext.fused_op(
        x,
        conv_transpose_weight,
        conv_transpose_bias,
        bias,
        output,
        scaling_factor,
        batch_size,
        in_channels,
        out_channels,
        in_height,
        in_width,
        out_height,
        out_width,
        kernel_size,
        conv_transpose_stride[0],
        conv_transpose_padding[0]
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
