# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_164831/code_0.py
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
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# CUDA kernel implementation
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_op_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ conv_weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ bias_tensor,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_height,
    int in_width,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    float scaling_factor
) {
    // Calculate output dimensions for conv transpose
    int out_height = (in_height - 1) * stride - 2 * padding + kernel_size + output_padding;
    int out_width = (in_width - 1) * stride - 2 * padding + kernel_size + output_padding;
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * out_height * out_width;
    
    if (tid >= total_elements) return;
    
    // Calculate indices
    int batch_idx = tid / (out_channels * out_height * out_width);
    int remaining = tid % (out_channels * out_height * out_width);
    int out_ch_idx = remaining / (out_height * out_width);
    remaining = remaining % (out_height * out_width);
    int out_h_idx = remaining / out_width;
    int out_w_idx = remaining % out_width;
    
    // Perform conv transpose operation
    float sum = 0.0f;
    
    // Loop over input spatial locations that contribute to this output location
    for (int in_ch = 0; in_ch < in_channels; in_ch++) {
        for (int k_h = 0; k_h < kernel_size; k_h++) {
            for (int k_w = 0; k_w < kernel_size; k_w++) {
                // Calculate corresponding input position
                int in_h = out_h_idx + padding - k_h;
                int in_w = out_w_idx + padding - k_w;
                
                // Check if this corresponds to a valid input position considering stride
                if (in_h >= 0 && in_h < in_height * stride && in_h % stride == 0 &&
                    in_w >= 0 && in_w < in_width * stride && in_w % stride == 0) {
                    
                    int orig_in_h = in_h / stride;
                    int orig_in_w = in_w / stride;
                    
                    if (orig_in_h < in_height && orig_in_w < in_width) {
                        int input_idx = batch_idx * (in_channels * in_height * in_width) + 
                                       in_ch * (in_height * in_width) + 
                                       orig_in_h * in_width + orig_in_w;
                                       
                        int weight_idx = out_ch_idx * (in_channels * kernel_size * kernel_size) +
                                        in_ch * (kernel_size * kernel_size) +
                                        k_h * kernel_size + k_w;
                        
                        sum += input[input_idx] * conv_weight[weight_idx];
                    }
                }
            }
        }
    }
    
    // Add convolution bias
    sum += conv_bias[out_ch_idx];
    
    // Add bias tensor
    sum += bias_tensor[out_ch_idx];
    
    // First clamp (0-1)
    sum = fmaxf(0.0f, fminf(1.0f, sum));
    
    // Scale by factor
    sum *= scaling_factor;
    
    // Second clamp (0-1)
    sum = fmaxf(0.0f, fminf(1.0f, sum));
    
    // Divide by scaling factor
    sum /= scaling_factor;
    
    // Store result
    output[tid] = sum;
}

void fused_op_forward(
    const float* input,
    const float* conv_weight,
    const float* conv_bias,
    const float* bias_tensor,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_height,
    int in_width,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    float scaling_factor,
    int blocks,
    int threads
) {
    fused_op_forward_kernel<<<blocks, threads>>>(
        input, conv_weight, conv_bias, bias_tensor, output,
        batch_size, in_channels, out_channels, in_height, in_width,
        kernel_size, stride, padding, output_padding, scaling_factor
    );
}
"""

# C++ binding code
cpp_source = r"""
#include <torch/extension.h>

// Forward declaration of the function in the .cu file
void fused_op_forward(
    const float* input,
    const float* conv_weight,
    const float* conv_bias,
    const float* bias_tensor,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_height,
    int in_width,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    float scaling_factor,
    int blocks,
    int threads
);

void fused_op_forward_torch(
    torch::Tensor input,
    torch::Tensor conv_weight,
    torch::Tensor conv_bias,
    torch::Tensor bias_tensor,
    torch::Tensor output,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_height,
    int in_width,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    float scaling_factor
) {
    int out_height = (in_height - 1) * stride - 2 * padding + kernel_size + output_padding;
    int out_width = (in_width - 1) * stride - 2 * padding + kernel_size + output_padding;
    int total_elements = batch_size * out_channels * out_height * out_width;
    
    int threads_per_block = 256;
    int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    fused_op_forward(
        input.data_ptr<float>(),
        conv_weight.data_ptr<float>(),
        conv_bias.data_ptr<float>(),
        bias_tensor.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        in_height,
        in_width,
        kernel_size,
        stride,
        padding,
        output_padding,
        scaling_factor,
        blocks,
        threads_per_block
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward_torch, "Fused operation forward pass");
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
    # Validate that groups is 1 and dilation is (1, 1) as our kernel doesn't support other values
    if conv_transpose_groups != 1:
        raise ValueError("Only groups=1 is supported")
    if conv_transpose_dilation != (1, 1) and conv_transpose_dilation != [1, 1] and conv_transpose_dilation != 1:
        raise ValueError("Only dilation=1 is supported")
    
    batch_size, in_channels, in_height, in_width = x.shape
    out_channels = conv_transpose_weight.shape[0]
    kernel_size = conv_transpose_weight.shape[2]  # Assuming square kernel
    
    # Handle stride, padding, output_padding as tuples or integers
    if isinstance(conv_transpose_stride, (list, tuple)):
        stride_h, stride_w = conv_transpose_stride[0], conv_transpose_stride[1]
    else:
        stride_h = stride_w = conv_transpose_stride
        
    if isinstance(conv_transpose_padding, (list, tuple)):
        padding_h, padding_w = conv_transpose_padding[0], conv_transpose_padding[1]
    else:
        padding_h = padding_w = conv_transpose_padding
        
    if isinstance(conv_transpose_output_padding, (list, tuple)):
        output_padding_h, output_padding_w = conv_transpose_output_padding[0], conv_transpose_output_padding[1]
    else:
        output_padding_h = output_padding_w = conv_transpose_output_padding
    
    # Validate that we have symmetric parameters as our kernel assumes this
    if stride_h != stride_w or padding_h != padding_w or output_padding_h != output_padding_w:
        raise ValueError("Only symmetric stride, padding and output_padding are supported")
    
    stride = stride_h
    padding = padding_h
    output_padding = output_padding_h
    
    # Calculate output dimensions
    out_height = (in_height - 1) * stride - 2 * padding + kernel_size + output_padding
    out_width = (in_width - 1) * stride - 2 * padding + kernel_size + output_padding
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, out_height, out_width, device=x.device, dtype=x.dtype)
    
    # Call fused operation
    fused_ext.fused_op(
        x, conv_transpose_weight, conv_transpose_bias, bias, output,
        batch_size, in_channels, out_channels, in_height, in_width,
        kernel_size, stride, padding, output_padding, scaling_factor
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
