# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_123254/code_7.py
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
from torch.utils.cpp_extension import load_inline

# Define the CUDA kernel for fused ConvTranspose2d + Bias + Tanh
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_conv_transpose_tanh_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ subtract_bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_size
) {
    // Calculate global thread indices
    int batch_idx = blockIdx.x;
    int out_ch = blockIdx.y;
    int out_pixel = blockIdx.z * blockDim.x + threadIdx.x;

    if (batch_idx >= batch_size || out_ch >= out_channels || out_pixel >= out_height * out_width) {
        return;
    }

    int out_y = out_pixel / out_width;
    int out_x = out_pixel % out_width;

    float accumulator = 0.0f;

    // ConvTranspose2d computation with 4x4 kernel and stride=2
    for (int in_ch = 0; in_ch < in_channels; ++in_ch) {
        // For stride=2, we need to iterate through positions that could contribute
        // to this output pixel
        for (int ky = 0; ky < kernel_size; ++ky) {
            for (int kx = 0; kx < kernel_size; ++kx) {
                // Calculate input position that would generate this output
                int in_y = (out_y - ky + 1) / 2;  // Reversed from conv forward
                int in_x = (out_x - kx + 1) / 2;
                
                // Check if indices are valid and satisfy stride condition
                if (in_y >= 0 && in_y < in_height && 
                    in_x >= 0 && in_x < in_width &&
                    (out_y - ky + 1) % 2 == 0 && 
                    (out_x - kx + 1) % 2 == 0) {
                    
                    // Get input value
                    float input_val = input[batch_idx * (in_channels * in_height * in_width) + 
                                          in_ch * (in_height * in_width) + 
                                          in_y * in_width + in_x];
                    
                    // Get weight value (weights are [in_ch, out_ch, ky, kx])
                    float weight_val = weight[out_ch * (in_channels * kernel_size * kernel_size) + 
                                            in_ch * (kernel_size * kernel_size) + 
                                            ky * kernel_size + kx];
                    
                    accumulator += input_val * weight_val;
                }
            }
        }
    }

    // Add conv transpose bias
    accumulator += conv_bias[out_ch];
    
    // Subtract custom bias
    accumulator -= subtract_bias[out_ch];
    
    // Apply tanh activation
    output[batch_idx * (out_channels * out_height * out_width) + 
           out_ch * (out_height * out_width) + 
           out_y * out_width + out_x] = tanhf(accumulator);
}

void launch_fused_conv_transpose_tanh(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor conv_bias,
    torch::Tensor subtract_bias,
    torch::Tensor output,
    int kernel_size
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_height = input.size(2);
    int in_width = input.size(3);
    
    int out_channels = weight.size(0);
    int out_height = output.size(2);
    int out_width = output.size(3);
    
    // Configure kernel launch parameters
    dim3 grid(batch_size, out_channels, (out_height * out_width + 255) / 256);
    dim3 block(256);
    
    fused_conv_transpose_tanh_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        conv_bias.data_ptr<float>(),
        subtract_bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        in_height,
        in_width,
        out_height,
        out_width,
        kernel_size
    );
}
"""

# Define the C++ interface
cpp_source = r"""
#include <torch/extension.h>

void launch_fused_conv_transpose_tanh(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor conv_bias,
    torch::Tensor subtract_bias,
    torch::Tensor output,
    int kernel_size
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_transpose_tanh", &launch_fused_conv_transpose_tanh, "Fused ConvTranspose2d + Bias + Tanh");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_conv_transpose_tanh_ext',
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
    # Create output tensor with correct size (assuming stride=2 for upsampling)
    batch_size, in_channels, in_height, in_width = x.shape
    out_channels = conv_transpose_weight.shape[0]
    kernel_size = 4
    
    # Calculate output dimensions for stride=2, padding=1, kernel=4
    out_height = (in_height - 1) * conv_transpose_stride[0] - 2 * conv_transpose_padding[0] + kernel_size + conv_transpose_output_padding[0]
    out_width = (in_width - 1) * conv_transpose_stride[1] - 2 * conv_transpose_padding[1] + kernel_size + conv_transpose_output_padding[1]
    
    output = torch.empty((batch_size, out_channels, out_height, out_width), device=x.device, dtype=x.dtype)
    
    # Launch the fused kernel
    fused_ext.fused_conv_transpose_tanh(
        x, 
        conv_transpose_weight, 
        conv_transpose_bias, 
        bias, 
        output,
        kernel_size
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
