# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_121229/code_3.py
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
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# CUDA kernel that fuses conv_transpose2d + subtract bias + tanh
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_conv_transpose_bias_tanh_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_height,
    int input_width,
    int output_height,
    int output_width,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    int groups,
    int dilation
) {
    // Grid-stride loop for better utilization
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
         idx < batch_size * out_channels * output_height * output_width; 
         idx += blockDim.x * gridDim.x) {
        
        int tmp = idx;
        int out_x = tmp % output_width; tmp /= output_width;
        int out_y = tmp % output_height; tmp /= output_height;
        int out_ch = tmp % out_channels; tmp /= out_channels;
        int batch = tmp;
        
        if (batch >= batch_size) return;
        
        int group_id = out_ch / (out_channels / groups);
        float sum = 0.0f;
        
        // Conv transpose with optimized loop
        int in_ch_start = group_id * (in_channels / groups);
        int in_ch_end = (group_id + 1) * (in_channels / groups);
        
        for (int in_ch = in_ch_start; in_ch < in_ch_end; in_ch++) {
            for (int k_y = 0; k_y < kernel_size; k_y++) {
                for (int k_x = 0; k_x < kernel_size; k_x++) {
                    // Calculate corresponding input position
                    int in_x = out_x + padding - k_x * dilation;
                    int in_y = out_y + padding - k_y * dilation;
                    
                    if (in_x % stride == 0 && in_y % stride == 0) {
                        in_x /= stride;
                        in_y /= stride;
                        
                        if (in_x >= 0 && in_x < input_width && in_y >= 0 && in_y < input_height) {
                            int input_idx = ((batch * in_channels + in_ch) * input_height + in_y) * input_width + in_x;
                            int weight_idx = ((in_ch - in_ch_start) * out_channels + out_ch) * kernel_size * kernel_size + 
                                           k_y * kernel_size + k_x;
                            sum += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
        
        // Add conv bias
        if (conv_bias != nullptr) {
            sum += conv_bias[out_ch];
        }
        
        // Subtract bias and apply tanh
        sum -= bias[out_ch];
        output[idx] = tanhf(sum);
    }
}

void fused_conv_transpose_bias_tanh_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& conv_bias,
    const torch::Tensor& bias,
    torch::Tensor& output,
    int stride,
    int padding,
    int output_padding,
    int groups,
    int dilation
) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int input_height = input.size(2);
    const int input_width = input.size(3);
    const int out_channels = weight.size(1);
    const int kernel_size = weight.size(2);
    
    const int output_height = (input_height - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1;
    const int output_width = (input_width - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1;
    
    const dim3 block(256);
    const dim3 grid((batch_size * out_channels * output_height * output_width + block.x - 1) / block.x);
    
    fused_conv_transpose_bias_tanh_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        conv_bias.data_ptr<float>(),
        bias.data_ptr<float>(),
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
        dilation
    );
}
"""

# C++ interface for Python binding
cpp_source = r"""
#include <torch/extension.h>

void fused_conv_transpose_bias_tanh_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& conv_bias,
    const torch::Tensor& bias,
    torch::Tensor& output,
    int stride,
    int padding,
    int output_padding,
    int groups,
    int dilation
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_conv_transpose_bias_tanh_forward, "Fused conv transpose, bias subtract, and tanh");
}
"""

# Compile the extension with optimization flags
fused_ext = load_inline(
    name='fused_conv_transpose_tanh',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math', '-lineinfo'],
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
    # Calculate output dimensions
    batch_size, in_channels, input_height, input_width = x.shape
    out_channels, _, kernel_size, _ = conv_transpose_weight.shape
    
    output_height = (input_height - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_dilation * (kernel_size - 1) + conv_transpose_output_padding + 1
    output_width = (input_width - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_dilation * (kernel_size - 1) + conv_transpose_output_padding + 1
    
    # Create output tensor
    output = torch.empty((batch_size, out_channels, output_height, output_width), device=x.device, dtype=x.dtype)
    
    # Call fused kernel
    fused_ext.fused_op(
        x, 
        conv_transpose_weight, 
        conv_transpose_bias, 
        bias,
        output,
        conv_transpose_stride,
        conv_transpose_padding,
        conv_transpose_output_padding,
        conv_transpose_groups,
        conv_transpose_dilation
    )
    
    return output

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
