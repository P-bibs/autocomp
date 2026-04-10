# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_160018/code_1.py
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
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# CUDA kernel that fuses all operations
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__device__ float gelu_impl(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

__global__ void fused_conv_transpose_activation_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float add_value,
    const float multiply_value,
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
    int output_padding,
    int groups,
    int dilation) {
    
    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * out_height * out_width;
    
    if (idx >= total_elements) return;
    
    // Decompose linear index into multidimensional indices
    int batch_idx = idx / (out_channels * out_height * out_width);
    int temp = idx % (out_channels * out_height * out_width);
    int out_ch = temp / (out_height * out_width);
    temp = temp % (out_height * out_width);
    int out_y = temp / out_width;
    int out_x = temp % out_width;
    
    // Calculate group index
    int group_idx = out_ch * groups / out_channels;
    int channels_per_group = in_channels / groups;
    int out_ch_per_group = out_channels / groups;
    
    // Conv transpose calculation
    float sum = (bias != nullptr) ? bias[out_ch] : 0.0f;
    
    // Calculate input range that contributes to this output pixel
    for (int k_y = 0; k_y < kernel_size; k_y++) {
        for (int k_x = 0; k_x < kernel_size; k_x++) {
            // Calculate corresponding input position
            int in_y = (out_y + padding - k_y * dilation) / stride;
            int in_x = (out_x + padding - k_x * dilation) / stride;
            
            // Check if input position is valid and aligns with stride
            if ((out_y + padding - k_y * dilation) % stride == 0 &&
                (out_x + padding - k_x * dilation) % stride == 0 &&
                in_y >= 0 && in_y < in_height &&
                in_x >= 0 && in_x < in_width) {
                
                // Weight indexing: [out_ch, in_ch/group, k_h, k_w]
                int weight_idx = out_ch * channels_per_group * kernel_size * kernel_size +
                                ((k_y * kernel_size + k_x) * channels_per_group) +
                                (out_ch % out_ch_per_group);
                
                // Input indexing: [batch, in_ch, in_h, in_w]
                int input_ch = group_idx * channels_per_group + (out_ch % out_ch_per_group);
                int input_idx = batch_idx * in_channels * in_height * in_width +
                               input_ch * in_height * in_width +
                               in_y * in_width + in_x;
                
                sum += input[input_idx] * weight[weight_idx];
            }
        }
    }
    
    // Apply all remaining operations: add -> min -> gelu -> multiply
    sum += add_value;
    sum = fminf(sum, 0.0f);
    sum = gelu_impl(sum);
    sum *= multiply_value;
    
    // Write output
    output[idx] = sum;
}

void fused_conv_transpose_activation_forward(
    const torch::Tensor input,
    const torch::Tensor weight,
    const torch::Tensor bias,
    float add_value,
    float multiply_value,
    torch::Tensor output,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    int groups,
    int dilation) {
    
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_height = input.size(2);
    int in_width = input.size(3);
    int out_channels = weight.size(0);
    int out_height = output.size(2);
    int out_width = output.size(3);
    
    // Configure kernel launch parameters
    int total_elements = batch_size * out_channels * out_height * out_width;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    
    fused_conv_transpose_activation_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        add_value,
        multiply_value,
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
        output_padding,
        groups,
        dilation
    );
}
"""

# C++ binding code
cpp_source = r"""
#include <torch/extension.h>

void fused_conv_transpose_activation_forward(
    const torch::Tensor input,
    const torch::Tensor weight,
    const torch::Tensor bias,
    float add_value,
    float multiply_value,
    torch::Tensor output,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    int groups,
    int dilation);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_transpose_activation", &fused_conv_transpose_activation_forward, "Fused ConvTranspose2d + Activation forward");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_conv_transpose_activation_ext',
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
    # Calculate output dimensions
    batch_size, in_channels, in_height, in_width = x.shape
    out_channels, _, kernel_size, _ = conv_transpose_weight.shape
    
    # Compute output spatial dimensions for conv_transpose2d
    out_height = (in_height - 1) * conv_transpose_stride + (kernel_size - 1) * conv_transpose_dilation + 1 - 2 * conv_transpose_padding + conv_transpose_output_padding
    out_width = (in_width - 1) * conv_transpose_stride + (kernel_size - 1) * conv_transpose_dilation + 1 - 2 * conv_transpose_padding + conv_transpose_output_padding
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, out_height, out_width, device=x.device, dtype=x.dtype)
    
    # Call fused kernel
    fused_ext.fused_conv_transpose_activation(
        x, 
        conv_transpose_weight, 
        conv_transpose_bias,
        float(add_value),
        float(multiply_value),
        output,
        kernel_size,
        conv_transpose_stride,
        conv_transpose_padding,
        conv_transpose_output_padding,
        conv_transpose_groups,
        conv_transpose_dilation
    )
    
    return output

# Parameters for testing
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
