# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_084802/code_1.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_weight', 'conv_bias', 'conv_stride', 'conv_padding', 'conv_dilation', 'conv_groups']
REQUIRED_FLAT_STATE_NAMES = ['conv_weight', 'conv_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a convolution, applies minimum operation, Tanh, and another Tanh.
    """

    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

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
    # State for conv (nn.Conv2d)
    if 'conv_weight' in flat_state:
        state_kwargs['conv_weight'] = flat_state['conv_weight']
    else:
        state_kwargs['conv_weight'] = getattr(model.conv, 'weight', None)
    if 'conv_bias' in flat_state:
        state_kwargs['conv_bias'] = flat_state['conv_bias']
    else:
        state_kwargs['conv_bias'] = getattr(model.conv, 'bias', None)
    state_kwargs['conv_stride'] = model.conv.stride
    state_kwargs['conv_padding'] = model.conv.padding
    state_kwargs['conv_dilation'] = model.conv.dilation
    state_kwargs['conv_groups'] = model.conv.groups
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

# Custom CUDA kernel for fused conv2d + min + tanh operations
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_conv_min_tanh_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    // Calculate output dimensions
    int out_height = (height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_width = (width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    // Thread indices - each thread handles one output spatial position
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int spatial_size = batch_size * out_height * out_width;
    
    if (tid >= spatial_size) return;
    
    // Decode thread index
    int batch_idx = tid / (out_height * out_width);
    int spatial_idx = tid % (out_height * out_width);
    int out_y = spatial_idx / out_width;
    int out_x = spatial_idx % out_width;
    
    // Calculate input starting position
    int in_y_start = out_y * stride - padding;
    int in_x_start = out_x * stride - padding;
    
    // Find minimum across all output channels for this spatial position
    float min_val = INFINITY;
    
    for (int out_ch = 0; out_ch < out_channels; out_ch++) {
        // Perform convolution for this output channel at current spatial position
        float conv_result = 0.0f;
        
        // Loop through kernel and input channels
        for (int ky = 0; ky < kernel_size; ky++) {
            for (int kx = 0; kx < kernel_size; kx++) {
                int in_y = in_y_start + ky * dilation;
                int in_x = in_x_start + kx * dilation;
                
                // Check bounds
                if (in_y >= 0 && in_y < height && in_x >= 0 && in_x < width) {
                    for (int in_ch = 0; in_ch < in_channels; in_ch++) {
                        int weight_idx = out_ch * in_channels * kernel_size * kernel_size + 
                                        in_ch * kernel_size * kernel_size + 
                                        ky * kernel_size + kx;
                        
                        int input_idx = batch_idx * in_channels * height * width + 
                                       in_ch * height * width + 
                                       in_y * width + in_x;
                                       
                        conv_result += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        
        // Add bias
        conv_result += bias[out_ch];
        
        // Update minimum
        if (conv_result < min_val) {
            min_val = conv_result;
        }
    }
    
    // Apply tanh twice
    float intermediate = tanhf(min_val);
    float final_result = tanhf(intermediate);
    
    // Store result - output has shape [batch_size, 1, out_height, out_width]
    int output_idx = batch_idx * out_height * out_width + out_y * out_width + out_x;
    output[output_idx] = final_result;
}

void fused_conv_min_tanh_forward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::Tensor& output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    // Calculate output dimensions
    int out_height = (height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_width = (width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    // Launch configuration
    int spatial_size = batch_size * out_height * out_width;
    int threads_per_block = 256;
    int blocks = (spatial_size + threads_per_block - 1) / threads_per_block;
    
    // Launch kernel
    fused_conv_min_tanh_kernel<<<blocks, threads_per_block>>>(
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
        dilation
    );
    
    cudaDeviceSynchronize();
}
"""

# C++ bindings
cpp_source = r"""
#include <torch/extension.h>

void fused_conv_min_tanh_forward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::Tensor& output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int kernel_size,
    int stride,
    int padding,
    int dilation
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_min_tanh", &fused_conv_min_tanh_forward, "Fused conv2d + min + tanh operation");
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
    conv_weight,
    conv_bias,
    conv_stride,
    conv_padding,
    conv_dilation,
    conv_groups,
):
    batch_size, in_channels, height, width = x.shape
    out_channels = conv_weight.shape[0]
    kernel_size = conv_weight.shape[2]
    
    # Calculate output spatial dimensions
    out_height = (height + 2 * conv_padding - conv_dilation * (kernel_size - 1) - 1) // conv_stride + 1
    out_width = (width + 2 * conv_padding - conv_dilation * (kernel_size - 1) - 1) // conv_stride + 1
    
    # Create output tensor with correct shape [batch_size, 1, out_height, out_width]
    output = torch.empty(batch_size, 1, out_height, out_width, dtype=x.dtype, device=x.device)
    
    # Call fused operation
    fused_ext.fused_conv_min_tanh(
        x, conv_weight, conv_bias, output,
        batch_size, in_channels, out_channels, height, width,
        kernel_size, conv_stride, conv_padding, conv_dilation
    )
    
    return output

batch_size = 128
in_channels = 16
out_channels = 64
height = width = 256
kernel_size = 3

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]
