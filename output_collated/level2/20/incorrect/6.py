# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_125325/code_1.py
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

__global__ void fused_conv_transpose3d_ops_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias_tensor,
    const float* __restrict__ conv_bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_depth,
    int input_height,
    int input_width,
    int output_depth,
    int output_height,
    int output_width,
    int kernel_size,
    int stride,
    int padding,
    int output_padding
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * output_depth * output_height * output_width;
    
    if (idx >= total_elements) return;
    
    // Calculate output indices
    int w_out = idx % output_width;
    int h_out = (idx / output_width) % output_height;
    int d_out = (idx / (output_width * output_height)) % output_depth;
    int c_out = (idx / (output_width * output_height * output_depth)) % out_channels;
    int b = idx / (output_width * output_height * output_depth * out_channels);
    
    // Calculate input region bounds
    int d_start = max(0, (d_out + output_padding - kernel_size + padding + stride - 1) / stride);
    int h_start = max(0, (h_out + output_padding - kernel_size + padding + stride - 1) / stride);
    int w_start = max(0, (w_out + output_padding - kernel_size + padding + stride - 1) / stride);
    
    int d_end = min(input_depth, (d_out + output_padding + padding) / stride + 1);
    int h_end = min(input_height, (h_out + output_padding + padding) / stride + 1);
    int w_end = min(input_width, (w_out + output_padding + padding) / stride + 1);
    
    float conv_result = 0.0f;
    
    // Conv transpose computation
    for (int d_in = d_start; d_in < d_end; d_in++) {
        for (int h_in = h_start; h_in < h_end; h_in++) {
            for (int w_in = w_start; w_in < w_end; w_in++) {
                int k_d = d_out + output_padding - (d_in * stride - padding);
                int k_h = h_out + output_padding - (h_in * stride - padding);
                int k_w = w_out + output_padding - (w_in * stride - padding);
                
                if (k_d >= 0 && k_d < kernel_size && 
                    k_h >= 0 && k_h < kernel_size && 
                    k_w >= 0 && k_w < kernel_size) {
                    
                    for (int c_in = 0; c_in < in_channels; c_in++) {
                        int input_idx = b * (in_channels * input_depth * input_height * input_width) +
                                       c_in * (input_depth * input_height * input_width) +
                                       d_in * (input_height * input_width) +
                                       h_in * input_width +
                                       w_in;
                                       
                        int weight_idx = c_out * (in_channels * kernel_size * kernel_size * kernel_size) +
                                        c_in * (kernel_size * kernel_size * kernel_size) +
                                        k_d * (kernel_size * kernel_size) +
                                        k_h * kernel_size +
                                        k_w;
                                        
                        conv_result += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }
    
    // Add conv bias if provided
    if (conv_bias != nullptr) {
        conv_result += conv_bias[c_out];
    }
    
    // Store original value
    float original_x = conv_result;
    
    // Perform fused operations: x = ((x + bias) + original_x) * original_x + original_x
    float x = conv_result;
    x += bias_tensor[c_out];  // Add bias tensor
    x += original_x;          // x = x + original_x
    x *= original_x;          // x = x * original_x
    x += original_x;          // x = x + original_x
    
    output[idx] = x;
}

void fused_conv_transpose3d_ops(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias_tensor,
    const c10::optional<at::Tensor>& conv_bias,
    at::Tensor& output,
    int kernel_size,
    int stride,
    int padding,
    int output_padding
) {
    const at::cuda::CUDAGuard device_guard(input.device());
    
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_depth = input.size(2);
    int input_height = input.size(3);
    int input_width = input.size(4);
    
    int out_channels = weight.size(1);
    int output_depth = output.size(2);
    int output_height = output.size(3);
    int output_width = output.size(4);
    
    int total_elements = batch_size * out_channels * output_depth * output_height * output_width;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    
    const float* conv_bias_ptr = conv_bias.has_value() ? conv_bias.value().data_ptr<float>() : nullptr;
    
    fused_conv_transpose3d_ops_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_tensor.data_ptr<float>(),
        conv_bias_ptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_depth,
        input_height,
        input_width,
        output_depth,
        output_height,
        output_width,
        kernel_size,
        stride,
        padding,
        output_padding
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_conv_transpose3d_ops(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias_tensor,
    const c10::optional<at::Tensor>& conv_bias,
    at::Tensor& output,
    int kernel_size,
    int stride,
    int padding,
    int output_padding
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_transpose3d_ops", &fused_conv_transpose3d_ops, "Fused ConvTranspose3d with operations");
}
"""

fused_ext = load_inline(
    name='fused_conv_transpose3d_ops_ext',
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
    # Ensure groups=1 and dilation=1 for our kernel
    assert conv_transpose_groups == 1, "Only groups=1 supported"
    assert conv_transpose_dilation == 1, "Only dilation=1 supported"
    
    batch_size = x.size(0)
    out_channels = conv_transpose_weight.size(1)
    kernel_size = conv_transpose_weight.size(2)
    
    # Calculate output dimensions for conv transpose
    output_depth = (x.size(2) - 1) * conv_transpose_stride + conv_transpose_output_padding - 2 * conv_transpose_padding + kernel_size
    output_height = (x.size(3) - 1) * conv_transpose_stride + conv_transpose_output_padding - 2 * conv_transpose_padding + kernel_size
    output_width = (x.size(4) - 1) * conv_transpose_stride + conv_transpose_output_padding - 2 * conv_transpose_padding + kernel_size
    
    output = torch.empty(batch_size, out_channels, output_depth, output_height, output_width, device=x.device, dtype=x.dtype)
    
    # Call fused CUDA kernel
    fused_ext.fused_conv_transpose3d_ops(
        x,
        conv_transpose_weight,
        bias,
        conv_transpose_bias if conv_transpose_bias is not None else None,
        output,
        kernel_size,
        conv_transpose_stride,
        conv_transpose_padding,
        conv_transpose_output_padding
    )
    
    return output

# Test parameters
batch_size = 16
in_channels = 32
out_channels = 64
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1, 1)

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape]

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]
