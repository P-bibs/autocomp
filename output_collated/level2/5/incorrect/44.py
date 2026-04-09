# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_121229/code_1.py
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

# Define the fused CUDA kernel
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

// CUDA kernel for fused transposed convolution + bias subtract + tanh
__global__ void fused_conv_transpose_tanh_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ tanh_bias,
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
    int output_padding
) {
    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = batch_size * out_channels * output_height * output_width;
    
    // Handle multiple iterations if we have fewer threads than elements
    for (int global_idx = idx; global_idx < total_threads; global_idx += blockDim.x * gridDim.x) {
        // Decompose global index into 4D coordinates
        int temp = global_idx;
        int out_w = temp % output_width;
        temp /= output_width;
        int out_h = temp % output_height;
        temp /= output_height;
        int out_c = temp % out_channels;
        int batch = temp / out_channels;
        
        // Calculate input position for this output pixel
        float sum = 0.0f;
        
        // Perform transposed convolution
        for (int in_c = 0; in_c < in_channels; in_c++) {
            for (int ky = 0; ky < kernel_size; ky++) {
                for (int kx = 0; kx < kernel_size; kx++) {
                    // Calculate corresponding input position
                    int in_x = out_w - kx * stride + padding;
                    int in_y = out_h - ky * stride + padding;
                    
                    // Check if within input bounds
                    if (in_x >= 0 && in_x < input_width && in_y >= 0 && in_y < input_height) {
                        // Calculate indices
                        int input_idx = ((batch * in_channels + in_c) * input_height + in_y) * input_width + in_x;
                        int weight_idx = (out_c * in_channels + in_c) * kernel_size * kernel_size + 
                                         (kernel_size - 1 - ky) * kernel_size + (kernel_size - 1 - kx);
                        
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        
        // Add convolution bias
        sum += conv_bias[out_c];
        
        // Apply tanh with bias subtraction
        float result = tanhf(sum - tanh_bias[out_c]);
        
        // Write to output
        int output_idx = ((batch * out_channels + out_c) * output_height + out_h) * output_width + out_w;
        output[output_idx] = result;
    }
}

// Host function to launch the kernel
void fused_conv_transpose_tanh_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor conv_bias,
    torch::Tensor tanh_bias,
    torch::Tensor output,
    int kernel_size,
    int stride,
    int padding,
    int output_padding
) {
    // Get tensor dimensions
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_height = input.size(2);
    int input_width = input.size(3);
    
    int out_channels = weight.size(0);
    int output_height = (input_height - 1) * stride - 2 * padding + kernel_size + output_padding;
    int output_width = (input_width - 1) * stride - 2 * padding + kernel_size + output_padding;
    
    // Calculate total number of output elements
    int total_elements = batch_size * out_channels * output_height * output_width;
    
    // Set up grid and block dimensions
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    blocks = min(blocks, 65536); // Limit blocks to avoid excessive kernel launches
    
    // Launch kernel
    fused_conv_transpose_tanh_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        conv_bias.data_ptr<float>(),
        tanh_bias.data_ptr<float>(),
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
        output_padding
    );
    
    // Check for errors
    TORCH_CHECK(cudaGetLastError() == cudaSuccess);
}
"""

# C++ binding code
cpp_source = r"""
#include <torch/extension.h>

void fused_conv_transpose_tanh_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor conv_bias,
    torch::Tensor tanh_bias,
    torch::Tensor output,
    int kernel_size,
    int stride,
    int padding,
    int output_padding
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_transpose_tanh_forward", &fused_conv_transpose_tanh_forward, 
          "Fused transposed convolution + bias subtract + tanh activation");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_conv_transpose_tanh',
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
):
    # Validate that we only support simple cases for this optimization
    if conv_transpose_groups != 1 or conv_transpose_dilation != (1, 1):
        raise ValueError("Only simple transposed convolutions supported in this optimization")
    
    # Create output tensor with correct shape
    stride = conv_transpose_stride[0]  # Assuming square and symmetric
    padding = conv_transpose_padding[0]
    output_padding = conv_transpose_output_padding[0]
    kernel_size = conv_transpose_weight.shape[2]  # Assuming square kernel
    
    output_height = (x.shape[2] - 1) * stride - 2 * padding + kernel_size + output_padding
    output_width = (x.shape[3] - 1) * stride - 2 * padding + kernel_size + output_padding
    output_shape = (x.shape[0], conv_transpose_weight.shape[0], output_height, output_width)
    output = torch.empty(output_shape, device=x.device, dtype=x.dtype)
    
    # Call the fused kernel
    fused_ext.fused_conv_transpose_tanh_forward(
        x, 
        conv_transpose_weight, 
        conv_transpose_bias, 
        bias.view(-1), 
        output,
        kernel_size,
        stride,
        padding,
        output_padding
    )
    
    return output

# Setup for testing
batch_size = 32
in_channels = 64
out_channels = 64
height = width = 256
kernel_size = 4
bias_shape = (out_channels, 1, 1)

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]
