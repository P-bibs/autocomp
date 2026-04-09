# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_090933/code_1.py
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

# CUDA kernel that fuses conv2d + min + tanh + tanh operations
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

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
    // Calculate output indices
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_b = blockIdx.z;
    
    // Get output dimensions
    int conv_out_h = (height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int conv_out_w = (width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    if (out_x >= conv_out_w || out_y >= conv_out_h || out_b >= batch_size) return;
    
    // For each output position, compute conv -> min -> tanh -> tanh
    float min_val = FLT_MAX;
    
    // Compute convolution for all output channels at this spatial location
    for (int out_c = 0; out_c < out_channels; out_c++) {
        float conv_sum = 0.0f;
        
        // Convolution computation
        for (int kc = 0; kc < in_channels; kc++) {
            for (int ky = 0; ky < kernel_size; ky++) {
                for (int kx = 0; kx < kernel_size; kx++) {
                    int in_y = out_y * stride - padding + ky * dilation;
                    int in_x = out_x * stride - padding + kx * dilation;
                    
                    if (in_y >= 0 && in_y < height && in_x >= 0 && in_x < width) {
                        int input_idx = out_b * (in_channels * height * width) + 
                                       kc * (height * width) + 
                                       in_y * width + in_x;
                        int weight_idx = out_c * (in_channels * kernel_size * kernel_size) + 
                                        kc * (kernel_size * kernel_size) + 
                                        ky * kernel_size + kx;
                        conv_sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        
        // Add bias
        conv_sum += bias[out_c];
        
        // Track minimum
        if (conv_sum < min_val) {
            min_val = conv_sum;
        }
    }
    
    // Apply double tanh
    float result = tanhf(min_val);
    result = tanhf(result);
    
    // Write output
    int output_idx = out_b * (1 * conv_out_h * conv_out_w) + 
                    0 * (conv_out_h * conv_out_w) + 
                    out_y * conv_out_w + out_x;
    output[output_idx] = result;
}

void fused_conv_min_tanh_forward(
    const torch::Tensor input,
    const torch::Tensor weight,
    const torch::Tensor bias,
    torch::Tensor output,
    int stride,
    int padding,
    int dilation,
    int groups
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);
    
    // Calculate output dimensions
    int conv_out_h = (height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int conv_out_w = (width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    // Launch configuration
    dim3 block(16, 16, 1);
    dim3 grid(
        (conv_out_w + block.x - 1) / block.x,
        (conv_out_h + block.y - 1) / block.y,
        batch_size
    );
    
    fused_conv_min_tanh_kernel<<<grid, block>>>(
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
}
"""

# C++ interface bindings
cpp_source = r"""
#include <torch/extension.h>

void fused_conv_min_tanh_forward(
    const torch::Tensor input,
    const torch::Tensor weight,
    const torch::Tensor bias,
    torch::Tensor output,
    int stride,
    int padding,
    int dilation,
    int groups
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_min_tanh", &fused_conv_min_tanh_forward, "Fused conv + min + tanh + tanh operation");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_conv_min_tanh_ext',
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
    # Calculate output dimensions
    batch_size, in_channels, height, width = x.shape
    out_channels = conv_weight.shape[0]
    kernel_size = conv_weight.shape[2]
    
    conv_out_h = (height + 2 * conv_padding - conv_dilation * (kernel_size - 1) - 1) // conv_stride + 1
    conv_out_w = (width + 2 * conv_padding - conv_dilation * (kernel_size - 1) - 1) // conv_stride + 1
    
    # Create output tensor
    output = torch.empty(batch_size, 1, conv_out_h, conv_out_w, device=x.device, dtype=x.dtype)
    
    # Call fused CUDA kernel
    fused_ext.fused_conv_min_tanh(
        x, conv_weight, conv_bias, output,
        conv_stride, conv_padding, conv_dilation, conv_groups
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
