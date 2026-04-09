# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_051208/code_0.py
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

# CUDA kernel that fuses convolution, hardswish, and relu
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_conv_act_kernel(
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
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    if (batch_idx >= batch_size || out_ch >= out_channels) return;
    
    // Each thread processes multiple output elements
    for (int out_y = tid; out_y < out_height; out_y += block_size) {
        for (int out_x = 0; out_x < out_width; out_x++) {
            float sum = 0.0f;
            
            // Convolution computation
            for (int in_ch = 0; in_ch < in_channels; in_ch++) {
                for (int k_y = 0; k_y < kernel_size; k_y++) {
                    for (int k_x = 0; k_x < kernel_size; k_x++) {
                        int in_y = out_y * stride - padding + k_y * dilation;
                        int in_x = out_x * stride - padding + k_x * dilation;
                        
                        float pixel_val = 0.0f;
                        if (in_y >= 0 && in_y < height && in_x >= 0 && in_x < width) {
                            int input_idx = batch_idx * (in_channels * height * width) + 
                                          in_ch * (height * width) + 
                                          in_y * width + in_x;
                            pixel_val = input[input_idx];
                        }
                        
                        int weight_idx = out_ch * (in_channels * kernel_size * kernel_size) + 
                                       in_ch * (kernel_size * kernel_size) + 
                                       k_y * kernel_size + k_x;
                        sum += pixel_val * weight[weight_idx];
                    }
                }
            }
            
            // Add bias
            sum += bias[out_ch];
            
            // Hardswish: x * relu6(x + 3) / 6
            float hardswish_val = sum * fminf(fmaxf(sum + 3.0f, 0.0f), 6.0f) / 6.0f;
            
            // Relu: max(0, x)
            float relu_val = fmaxf(hardswish_val, 0.0f);
            
            // Store result
            int output_idx = batch_idx * (out_channels * out_height * out_width) + 
                           out_ch * (out_height * out_width) + 
                           out_y * out_width + out_x;
            output[output_idx] = relu_val;
        }
    }
}

void fused_conv_act_forward(
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
    
    // Configure kernel launch parameters
    dim3 grid(batch_size, out_channels);
    dim3 block(256); // 256 threads per block for good occupancy on RTX 2080Ti
    
    fused_conv_act_kernel<<<grid, block>>>(
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

# C++ interface/bindings
cpp_source = r"""
#include <torch/extension.h>

void fused_conv_act_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    int stride,
    int padding,
    int dilation
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_act", &fused_conv_act_forward, "Fused convolution with activations");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_conv_act',
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
    # Ensure inputs are on CUDA
    if not x.is_cuda:
        x = x.cuda()
    if not conv_weight.is_cuda:
        conv_weight = conv_weight.cuda()
    if not conv_bias.is_cuda:
        conv_bias = conv_bias.cuda()
    
    # Calculate output dimensions
    batch_size, in_channels, height, width = x.shape
    out_channels = conv_weight.shape[0]
    kernel_size = conv_weight.shape[2]
    
    out_height = (height + 2 * conv_padding - conv_dilation * (kernel_size - 1) - 1) // conv_stride + 1
    out_width = (width + 2 * conv_padding - conv_dilation * (kernel_size - 1) - 1) // conv_stride + 1
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, out_height, out_width, device=x.device, dtype=x.dtype)
    
    # Call fused CUDA kernel
    fused_ext.fused_conv_act(x, conv_weight, conv_bias, output, conv_stride, conv_padding, conv_dilation)
    
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
