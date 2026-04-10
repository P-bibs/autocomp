# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_164831/code_1.py
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

# Optimization: CUDA C++ Implementation of a fused Transposed Convolution pipeline
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define BLOCK_SIZE 16

__global__ void fused_conv_tr_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ add_bias,
    float scaling_factor,
    int batch_size,
    int in_channels,
    int in_height,
    int in_width,
    int out_channels,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    int out_height,
    int out_width
) {
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oc = blockIdx.z;

    if (oh >= out_height || ow >= out_width) return;

    float val = 0.0f;
    
    // ConvTranspose2d computation
    for (int ic = 0; ic < in_channels; ++ic) {
        int kh_start = fmaxf(0, (oh + padding - (out_height - 1) * stride + in_height * stride - 1) / stride);
        int kh_end = fminf(kernel_size, (oh + padding + stride - 1) / stride + 1);
        int kw_start = fmaxf(0, (ow + padding - (out_width - 1) * stride + in_width * stride - 1) / stride);
        int kw_end = fminf(kernel_size, (ow + padding + stride - 1) / stride + 1);
        
        for (int kh = kh_start; kh < kh_end; ++kh) {
            for (int kw = kw_start; kw < kw_end; ++kw) {
                int ih = oh + padding - kh * stride;
                int iw = ow + padding - kw * stride;
                
                if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                    int input_idx = ((blockIdx.z * batch_size + threadIdx.z) * in_channels + ic) * in_height * in_width + ih * in_width + iw;
                    int weight_idx = ic * out_channels * kernel_size * kernel_size + oc * kernel_size * kernel_size + (kernel_size - 1 - kh) * kernel_size + (kernel_size - 1 - kw);
                    val += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    // Add convolution bias
    val += conv_bias[oc];

    // Add additional bias
    val += add_bias[oc];

    // First clamp
    val = fminf(fmaxf(val, 0.0f), 1.0f);

    // Scale
    val *= scaling_factor;

    // Second clamp
    val = fminf(fmaxf(val, 0.0f), 1.0f);

    // Divide by scaling factor
    val /= scaling_factor;

    int output_idx = ((blockIdx.z * batch_size + threadIdx.z) * out_channels + oc) * out_height * out_width + oh * out_width + ow;
    output[output_idx] = val;
}

void fused_op_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor conv_bias,
    torch::Tensor add_bias,
    torch::Tensor output,
    float scaling_factor,
    int stride,
    int padding,
    int output_padding
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_height = input.size(2);
    int in_width = input.size(3);
    
    int out_channels = weight.size(1);
    int kernel_size = weight.size(2);
    
    int out_height = (in_height - 1) * stride - 2 * padding + kernel_size + output_padding;
    int out_width = (in_width - 1) * stride - 2 * padding + kernel_size + output_padding;
    
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks(
        (out_width + threads.x - 1) / threads.x,
        (out_height + threads.y - 1) / threads.y,
        out_channels
    );
    
    fused_conv_tr_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        weight.data_ptr<float>(),
        conv_bias.data_ptr<float>(),
        add_bias.data_ptr<float>(),
        scaling_factor,
        batch_size,
        in_channels,
        in_height,
        in_width,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        out_height,
        out_width
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor conv_bias,
    torch::Tensor add_bias,
    torch::Tensor output,
    float scaling_factor,
    int stride,
    int padding,
    int output_padding
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused ConvTranspose2d with bias add and clamp operations");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_ct_op',
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
    scaling_factor,
):
    # Calculate output dimensions
    in_height, in_width = x.shape[2], x.shape[3]
    kernel_size = conv_transpose_weight.shape[2]
    out_height = (in_height - 1) * conv_transpose_stride - 2 * conv_transpose_padding + kernel_size + conv_transpose_output_padding
    out_width = (in_width - 1) * conv_transpose_stride - 2 * conv_transpose_padding + kernel_size + conv_transpose_output_padding
    out_channels = conv_transpose_weight.shape[1]
    
    # Create output tensor
    output = torch.empty((x.shape[0], out_channels, out_height, out_width), device=x.device, dtype=x.dtype)
    
    # Call fused CUDA kernel
    fused_ext.fused_op(
        x,
        conv_transpose_weight,
        conv_transpose_bias,
        bias,
        output,
        scaling_factor,
        conv_transpose_stride,
        conv_transpose_padding,
        conv_transpose_output_padding
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
