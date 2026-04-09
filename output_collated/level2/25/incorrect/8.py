# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_082041/code_5.py
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

# CUDA kernel for fused convolution + min + tanh
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>
#include <limits>

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

__global__ void fused_conv2d_min_tanh_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_height,
    int input_width,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w,
    int groups,
    int output_height,
    int output_width
) {
    // Calculate output position
    int out_w = blockIdx.x * blockDim.x + threadIdx.x;
    int out_h = blockIdx.y * blockDim.y + threadIdx.y;
    int batch_idx = blockIdx.z;
    
    if (out_w >= output_width || out_h >= output_height || batch_idx >= batch_size) {
        return;
    }
    
    int group_out_channels = out_channels / groups;
    int group_in_channels = in_channels / groups;
    
    float min_val = INFINITY;
    
    // Iterate through all output channels to find minimum
    for (int out_ch = 0; out_ch < out_channels; out_ch++) {
        int group_id = out_ch / group_out_channels;
        int group_start_in_ch = group_id * group_in_channels;
        
        float conv_sum = 0.0f;
        if (bias != NULL) {
            conv_sum = bias[out_ch];
        }
        
        // Perform convolution for this output channel
        for (int in_ch = 0; in_ch < group_in_channels; in_ch++) {
            int actual_in_ch = group_start_in_ch + in_ch;
            
            for (int kh = 0; kh < kernel_h; kh++) {
                for (int kw = 0; kw < kernel_w; kw++) {
                    int in_h = out_h * stride_h - padding_h + kh * dilation_h;
                    int in_w = out_w * stride_w - padding_w + kw * dilation_w;
                    
                    if (in_h >= 0 && in_h < input_height && in_w >= 0 && in_w < input_width) {
                        int input_idx = batch_idx * (in_channels * input_height * input_width) +
                                        actual_in_ch * (input_height * input_width) +
                                        in_h * input_width + in_w;
                                        
                        int weight_idx = out_ch * (group_in_channels * kernel_h * kernel_w) +
                                         in_ch * (kernel_h * kernel_w) +
                                         kh * kernel_w + kw;
                                         
                        conv_sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        
        // Update minimum
        min_val = fminf(min_val, conv_sum);
    }
    
    // Apply tanh activation
    float result = tanhf(min_val);
    
    // Write output (single channel due to min reduction)
    int output_idx = batch_idx * (1 * output_height * output_width) +
                     0 * (output_height * output_width) +
                     out_h * output_width + out_w;
                     
    output[output_idx] = result;
}

void fused_conv2d_min_tanh_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w,
    int groups
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_height = input.size(2);
    int input_width = input.size(3);
    int out_channels = weight.size(0);
    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);
    
    // Calculate output dimensions
    int output_height = (input_height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    int output_width = (input_width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
    
    // Configure grid and block dimensions
    dim3 block_size(16, 16, 1);
    dim3 grid_size(
        CEIL_DIV(output_width, block_size.x),
        CEIL_DIV(output_height, block_size.y),
        batch_size
    );
    
    float* bias_ptr = bias.defined() && bias.numel() > 0 ? bias.data_ptr<float>() : NULL;
    
    fused_conv2d_min_tanh_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_height,
        input_width,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
        groups,
        output_height,
        output_width
    );
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("CUDA kernel launch failed: ", cudaGetErrorString(err));
    }
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_conv2d_min_tanh_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w,
    int groups
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv2d_min_tanh", &fused_conv2d_min_tanh_forward, "Fused conv2d + min + tanh operation");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_conv2d_min_tanh',
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
    # Handle different types of stride/padding/dilation parameters
    if isinstance(conv_stride, int):
        stride_h = stride_w = conv_stride
    else:
        stride_h, stride_w = conv_stride
    
    if isinstance(conv_padding, int):
        padding_h = padding_w = conv_padding
    else:
        padding_h, padding_w = conv_padding
        
    if isinstance(conv_dilation, int):
        dilation_h = dilation_w = conv_dilation
    else:
        dilation_h, dilation_w = conv_dilation
    
    # Ensure inputs are contiguous
    x = x.contiguous()
    conv_weight = conv_weight.contiguous()
    if conv_bias is not None:
        conv_bias = conv_bias.contiguous()
    else:
        conv_bias = torch.empty(0, dtype=x.dtype, device=x.device)
    
    # Calculate output dimensions
    input_height, input_width = x.shape[2], x.shape[3]
    kernel_h, kernel_w = conv_weight.shape[2], conv_weight.shape[3]
    
    output_height = (input_height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
    output_width = (input_width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1
    
    # Create output tensor (1 channel due to min reduction)
    x_out = torch.empty(x.shape[0], 1, output_height, output_width, dtype=x.dtype, device=x.device)
    
    # Call fused kernel
    fused_ext.fused_conv2d_min_tanh(
        x,
        conv_weight,
        conv_bias,
        x_out,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
        conv_groups
    )
    
    return x_out

batch_size = 128
in_channels = 16
out_channels = 64
height = width = 256
kernel_size = 3

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]
