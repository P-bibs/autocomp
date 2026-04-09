# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_010751/code_0.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'scale_factor']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_weight', 'conv_bias', 'conv_stride', 'conv_padding', 'conv_dilation', 'conv_groups', 'scale_factor']
REQUIRED_FLAT_STATE_NAMES = ['conv_weight', 'conv_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a convolution, scales the output, and then applies a minimum operation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor

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
    if 'scale_factor' in flat_state:
        state_kwargs['scale_factor'] = flat_state['scale_factor']
    else:
        state_kwargs['scale_factor'] = getattr(model, 'scale_factor')
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

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

__global__ void fused_conv_scale_min_kernel(
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
    float scale_factor
) {
    int batch_idx = blockIdx.x;
    int out_h = blockIdx.y;
    int out_w = blockIdx.z;
    
    if (batch_idx >= batch_size || out_h >= height || out_w >= width) {
        return;
    }
    
    // Calculate output dimensions after convolution
    int out_height = (height + 2 * padding - kernel_size) / stride + 1;
    int out_width = (width + 2 * padding - kernel_size) / stride + 1;
    
    if (out_h >= out_height || out_w >= out_width) {
        return;
    }
    
    float global_min = 1e38f;
    
    // For each output channel, compute conv -> scale -> find min
    for (int out_ch = 0; out_ch < out_channels; ++out_ch) {
        float conv_sum = 0.0f;
        
        // Convolution
        for (int in_ch = 0; in_ch < in_channels; ++in_ch) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    int in_h = out_h * stride + kh - padding;
                    int in_w = out_w * stride + kw - padding;
                    
                    if (in_h >= 0 && in_h < height && in_w >= 0 && in_w < width) {
                        int input_idx = batch_idx * (in_channels * height * width) + 
                                       in_ch * (height * width) + 
                                       in_h * width + in_w;
                        int weight_idx = out_ch * (in_channels * kernel_size * kernel_size) +
                                        in_ch * (kernel_size * kernel_size) +
                                        kh * kernel_size + kw;
                        conv_sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        
        // Add bias
        conv_sum += bias[out_ch];
        
        // Apply scale factor
        conv_sum *= scale_factor;
        
        // Update global minimum
        global_min = fminf(global_min, conv_sum);
    }
    
    // Write result
    int output_idx = batch_idx * (1 * out_height * out_width) + 
                    0 * (out_height * out_width) + 
                    out_h * out_width + out_w;
    output[output_idx] = global_min;
}

void fused_conv_scale_min_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int stride,
    int padding,
    float scale_factor
) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);
    
    // Grid and block dimensions
    dim3 grid(batch_size, height, width);
    dim3 block(1, 1, 1);  // Simplified for now
    
    fused_conv_scale_min_kernel<<<grid, block>>>(
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
        scale_factor
    );
}
"""

# --- C++ Logic (Interface/Bindings) ---
cpp_source = r"""
#include <torch/extension.h>

void fused_conv_scale_min_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int stride,
    int padding,
    float scale_factor
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_scale_min_forward", &fused_conv_scale_min_forward, "Fused Conv + Scale + Min forward");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_conv_scale_min',
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
    scale_factor,
):
    # Ensure dilation=1 and groups=1 for this simplified implementation
    assert conv_dilation == 1 and conv_groups == 1, "Only dilation=1 and groups=1 supported"
    
    batch_size = x.shape[0]
    out_channels = conv_weight.shape[0]
    in_channels = conv_weight.shape[1]
    height = x.shape[2]
    width = x.shape[3]
    kernel_size = conv_weight.shape[2]
    
    # Calculate output dimensions
    out_height = (height + 2 * conv_padding - kernel_size) // conv_stride + 1
    out_width = (width + 2 * conv_padding - kernel_size) // conv_stride + 1
    
    # Create output tensor with correct shape (keepdim=True results in 1 channel)
    output = torch.empty(batch_size, 1, out_height, out_width, device=x.device, dtype=x.dtype)
    
    # Call fused CUDA kernel
    fused_ext.fused_conv_scale_min_forward(
        x, conv_weight, conv_bias, output, conv_stride, conv_padding, scale_factor
    )
    
    return output

batch_size = 64
in_channels = 64
out_channels = 128
height = width = 256
kernel_size = 3
scale_factor = 2.0

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scale_factor]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]
