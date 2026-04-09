# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_050250/code_3.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_weight', 'conv_bias', 'conv_stride', 'conv_padding', 'conv_dilation', 'conv_groups']
REQUIRED_FLAT_STATE_NAMES = ['conv_weight', 'conv_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a convolution, applies HardSwish, and then ReLU.
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

# CUDA kernel for fused convolution + hardswish + relu
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_conv_activation_kernel(
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
    
    // Get thread indices
    int batch_idx = blockIdx.x;
    int out_ch = blockIdx.y;
    int out_y = threadIdx.y + blockIdx.z * blockDim.y;
    int out_x = threadIdx.x + blockIdx.z * blockDim.z * blockDim.y;
    
    if (batch_idx >= batch_size || out_ch >= out_channels || out_y >= out_height || out_x >= out_width) {
        return;
    }
    
    // Shared memory for input patch
    extern __shared__ float shared_input[];
    
    // Calculate output position
    int out_idx = batch_idx * (out_channels * out_height * out_width) +
                  out_ch * (out_height * out_width) +
                  out_y * out_width + out_x;
    
    // Perform convolution
    float sum = 0.0f;
    
    // Add bias first
    if (bias != nullptr) {
        sum = bias[out_ch];
    }
    
    // Loop over input channels and kernel
    for (int in_ch = 0; in_ch < in_channels; ++in_ch) {
        for (int ky = 0; ky < kernel_size; ++ky) {
            for (int kx = 0; kx < kernel_size; ++kx) {
                // Calculate input position
                int in_y = out_y * stride - padding + ky * dilation;
                int in_x = out_x * stride - padding + kx * dilation;
                
                // Check bounds
                if (in_y >= 0 && in_y < height && in_x >= 0 && in_x < width) {
                    // Calculate indices
                    int input_idx = batch_idx * (in_channels * height * width) +
                                    in_ch * (height * width) +
                                    in_y * width + in_x;
                                    
                    int weight_idx = out_ch * (in_channels * kernel_size * kernel_size) +
                                     in_ch * (kernel_size * kernel_size) +
                                     ky * kernel_size + kx;
                    
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    // Apply hardswish activation: x * relu6(x + 3) / 6
    float hardswish_result;
    if (sum <= -3.0f) {
        hardswish_result = 0.0f;
    } else if (sum >= 3.0f) {
        hardswish_result = sum;
    } else {
        hardswish_result = sum * (sum + 3.0f) / 6.0f;
    }
    
    // Apply relu activation
    float final_result = fmaxf(0.0f, hardswish_result);
    
    // Write output
    output[out_idx] = final_result;
}

void fused_conv_activation_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    int stride,
    int padding,
    int dilation
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);
    
    int out_height = (height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_width = (width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    // Launch configuration
    dim3 block(1, 8, 8);  // 64 threads per block
    dim3 grid(batch_size, out_channels, (out_height * out_width + 63) / 64);
    
    // Shared memory size (for a patch of input)
    size_t shared_mem_size = in_channels * kernel_size * kernel_size * sizeof(float);
    
    fused_conv_activation_kernel<<<grid, block, shared_mem_size>>>(
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

# C++ interface
cpp_source = r"""
#include <torch/extension.h>

void fused_conv_activation_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    int stride,
    int padding,
    int dilation
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_activation", &fused_conv_activation_forward, "Fused Convolution + Hardswish + ReLU forward");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_conv_activation',
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
    # Only support conv_groups=1 for this implementation
    if conv_groups != 1:
        raise NotImplementedError("Only conv_groups=1 is supported")
        
    # Calculate output dimensions
    batch_size, in_channels, height, width = x.shape
    out_channels = conv_weight.shape[0]
    kernel_size = conv_weight.shape[2]
    
    out_height = (height + 2 * conv_padding - conv_dilation * (kernel_size - 1) - 1) // conv_stride + 1
    out_width = (width + 2 * conv_padding - conv_dilation * (kernel_size - 1) - 1) // conv_stride + 1
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, out_height, out_width, device=x.device, dtype=x.dtype)
    
    # Call fused kernel
    fused_ext.fused_conv_activation(
        x, conv_weight, conv_bias, output,
        conv_stride, conv_padding, conv_dilation
    )
    
    return output

batch_size = 128
in_channels = 8
out_channels = 64
height, width = 128, 128
kernel_size = 3

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]
