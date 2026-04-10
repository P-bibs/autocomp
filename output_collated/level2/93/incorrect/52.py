# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_160018/code_3.py
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
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fused ConvTranspose2d + Add + Min + Gelu + Multiply
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// GELU approximation
__device__ inline float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

// ConvTranspose2d + Post-processing fused kernel
__global__ void fused_conv_transpose_post_op_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    int groups,
    float add_value,
    float multiply_value
) {
    // Each thread handles one output element
    int total_output_elements = batch_size * out_channels * 
                                (height * stride - 2 * padding + kernel_size - 1 + output_padding) * 
                                (width * stride - 2 * padding + kernel_size - 1 + output_padding);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= total_output_elements) return;
    
    // Compute output indices
    int out_w = (width * stride - 2 * padding + kernel_size - 1 + output_padding);
    int out_h = (height * stride - 2 * padding + kernel_size - 1 + output_padding);
    int out_c = out_channels;
    
    int tmp = idx;
    int out_x = tmp % out_w; tmp /= out_w;
    int out_y = tmp % out_h; tmp /= out_h;
    int out_ch = tmp % out_c; tmp /= out_c;
    int batch_idx = tmp;
    
    if (batch_idx >= batch_size) return;
    
    // Perform convolution accumulation
    float sum = 0.0f;
    
    // Group handling (assuming groups divide in_channels and out_channels evenly)
    int group_idx = out_ch / (out_channels / groups);
    int in_ch_start = group_idx * (in_channels / groups);
    int in_ch_end = in_ch_start + (in_channels / groups);
    
    for (int in_ch = in_ch_start; in_ch < in_ch_end; ++in_ch) {
        for (int ky = 0; ky < kernel_size; ++ky) {
            for (int kx = 0; kx < kernel_size; ++kx) {
                // Compute input position
                int in_x = out_x - kx + padding;
                int in_y = out_y - ky + padding;
                
                // Check if input position is divisible by stride (transposed convolution property)
                if (in_x % stride == 0 && in_y % stride == 0) {
                    in_x /= stride;
                    in_y /= stride;
                    
                    // Bounds check
                    if (in_x >= 0 && in_x < width && in_y >= 0 && in_y < height) {
                        int input_idx = batch_idx * (in_channels * height * width) + 
                                       in_ch * (height * width) + 
                                       in_y * width + in_x;
                                       
                        int weight_idx = out_ch * (in_channels * kernel_size * kernel_size) + 
                                        in_ch * (kernel_size * kernel_size) + 
                                        (kernel_size - 1 - ky) * kernel_size + (kernel_size - 1 - kx);
                                        
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }
    
    // Add bias
    sum += bias[out_ch];
    
    // Post-processing operations
    sum += add_value;                           // Add
    sum = (sum < 0.0f) ? sum : 0.0f;            // Min with 0
    sum = gelu(sum);                            // GELU activation
    sum *= multiply_value;                      // Multiply
    
    output[idx] = sum;
}

void fused_conv_transpose_post_op(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    int groups,
    float add_value,
    float multiply_value
) {
    int out_height = height * stride - 2 * padding + kernel_size - 1 + output_padding;
    int out_width = width * stride - 2 * padding + kernel_size - 1 + output_padding;
    int total_elements = batch_size * out_channels * out_height * out_width;
    
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    
    fused_conv_transpose_post_op_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        kernel_size,
        stride,
        padding,
        output_padding,
        groups,
        add_value,
        multiply_value
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_conv_transpose_post_op(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    int groups,
    float add_value,
    float multiply_value
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_transpose_post_op", &fused_conv_transpose_post_op, "Fused ConvTranspose2d with post-processing operations");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_conv_transpose_op',
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
    add_value,
    multiply_value,
):
    # Get tensor dimensions
    batch_size, in_channels, height, width = x.shape
    out_channels = conv_transpose_weight.shape[0]
    kernel_size = conv_transpose_weight.shape[2]
    
    # Ensure dilation is (1,1) as our kernel doesn't support dilation
    if conv_transpose_dilation != (1, 1):
        raise NotImplementedError("Dilation not supported in this implementation")
    
    # Calculate output dimensions for ConvTranspose2d
    out_height = (height - 1) * conv_transpose_stride[0] - 2 * conv_transpose_padding[0] + kernel_size + conv_transpose_output_padding[0]
    out_width = (width - 1) * conv_transpose_stride[1] - 2 * conv_transpose_padding[1] + kernel_size + conv_transpose_output_padding[1]
    
    # Create output tensor
    output = torch.empty((batch_size, out_channels, out_height, out_width), device=x.device, dtype=x.dtype)
    
    # Ensure tensors are contiguous
    x = x.contiguous()
    conv_transpose_weight = conv_transpose_weight.contiguous()
    conv_transpose_bias = conv_transpose_bias.contiguous()
    
    # Call the fused kernel
    fused_ext.fused_conv_transpose_post_op(
        x,
        conv_transpose_weight,
        conv_transpose_bias,
        output,
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        kernel_size,
        conv_transpose_stride[0],  # Assuming square stride
        conv_transpose_padding[0],  # Assuming square padding
        conv_transpose_output_padding[0],  # Assuming square output padding
        conv_transpose_groups,
        float(add_value),
        float(multiply_value)
    )
    
    return output

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
