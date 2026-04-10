# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_160727/code_0.py
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
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# The CUDA source includes both conv_transpose2d and the fused operations
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__device__ __forceinline__ float fast_gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

__global__ void fused_conv_transpose2d_gelu_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
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
    float add_val,
    float mul_val
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = batch_size * out_channels * out_height * out_width;
    
    if (out_idx >= total_outputs) return;
    
    // Calculate indices
    int batch = out_idx / (out_channels * out_height * out_width);
    int remaining = out_idx % (out_channels * out_height * out_width);
    int out_ch = remaining / (out_height * out_width);
    remaining = remaining % (out_height * out_width);
    int out_y = remaining / out_width;
    int out_x = remaining % out_width;
    
    float result = 0.0f;
    
    // Perform transposed convolution calculation
    // For each position in the output, accumulate contributions from input positions
    for (int in_ch = 0; in_ch < in_channels; in_ch++) {
        for (int ky = 0; ky < kernel_size; ky++) {
            for (int kx = 0; kx < kernel_size; kx++) {
                // Calculate which input position would contribute to this output position
                int in_y = (out_y + padding - ky);
                int in_x = (out_x + padding - kx);
                
                // Check if this input position is valid after accounting for stride
                if (in_y >= 0 && in_y < in_height * stride && in_x >= 0 && in_x < in_width * stride) {
                    if (in_y % stride == 0 && in_x % stride == 0) {
                        int actual_in_y = in_y / stride;
                        int actual_in_x = in_x / stride;
                        
                        if (actual_in_y < in_height && actual_in_x < in_width) {
                            int input_idx = batch * (in_channels * in_height * in_width) + 
                                           in_ch * (in_height * in_width) + 
                                           actual_in_y * in_width + actual_in_x;
                            // Note: weight is organized as [in_channels, out_channels, kH, kW] 
                            // for transposed conv when viewed from forward perspective
                            int weight_idx = in_ch * (out_channels * kernel_size * kernel_size) + 
                                            out_ch * (kernel_size * kernel_size) + 
                                            ky * kernel_size + kx;
                            
                            result += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
    }
    
    // Add bias
    result += bias[out_ch];
    
    // Apply fused operations: add -> min -> gelu -> multiply
    float val = result + add_val;
    val = fminf(val, 0.0f);
    val = fast_gelu(val);
    output[out_idx] = val * mul_val;
}

void fused_conv_transpose2d_gelu_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int kernel_size,
    int stride,
    int padding,
    float add_val,
    float mul_val
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_height = input.size(2);
    int in_width = input.size(3);
    int out_height = output.size(2);
    int out_width = output.size(3);
    int out_channels = output.size(1);
    
    int total_outputs = batch_size * out_channels * out_height * out_width;
    int threads = 256;
    int blocks = (total_outputs + threads - 1) / threads;
    
    fused_conv_transpose2d_gelu_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
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
        add_val,
        mul_val
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_conv_transpose2d_gelu_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int kernel_size,
    int stride,
    int padding,
    float add_val,
    float mul_val
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_transpose2d_gelu", &fused_conv_transpose2d_gelu_forward, 
          "Fused conv_transpose2d + add-min-gelu-mul operation");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_conv_op',
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
    # Calculate output dimensions (matching PyTorch's conv_transpose2d formula)
    batch_size = x.size(0)
    in_channels = x.size(1)
    in_height = x.size(2)
    in_width = x.size(3)
    
    # Get kernel size from weight tensor
    kernel_size = conv_transpose_weight.size(2)  # assuming square kernel
    
    # Calculate output dimensions using PyTorch's formula for conv_transpose2d
    out_height = (in_height - 1) * conv_transpose_stride - 2 * conv_transpose_padding + kernel_size + conv_transpose_output_padding
    out_width = (in_width - 1) * conv_transpose_stride - 2 * conv_transpose_padding + kernel_size + conv_transpose_output_padding
    out_channels = conv_transpose_weight.size(0)
    
    # Create output tensor
    out = torch.empty(batch_size, out_channels, out_height, out_width, device='cuda', dtype=x.dtype)
    
    # Perform fused conv_transpose2d + operations
    fused_ext.fused_conv_transpose2d_gelu(
        x, 
        conv_transpose_weight, 
        conv_transpose_bias,
        out,
        kernel_size,
        conv_transpose_stride,
        conv_transpose_padding,
        float(add_value),
        float(multiply_value)
    )
    
    return out

# Constants provided by original context
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
    return [torch.rand(batch_size, in_channels, height, width, device='cuda')]
