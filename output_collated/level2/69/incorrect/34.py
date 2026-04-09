# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_052603/code_2.py
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

# CUDA kernel for fused Conv2d + HardSwish + ReLU
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__device__ float hardswish_impl(float x) {
    float relu6_val = fminf(fmaxf(x + 3.0f, 0.0f), 6.0f);
    return x * relu6_val / 6.0f;
}

__global__ void fused_conv_hardswish_relu_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int in_height,
    int in_width,
    int out_channels,
    int out_height,
    int out_width,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    int groups
) {
    int out_ch = blockIdx.x;
    int batch_idx = blockIdx.y;
    int out_y = threadIdx.x + blockIdx.z * blockDim.x;
    int out_x = threadIdx.y + blockIdx.z * blockDim.y * blockDim.x / out_width;
    
    // Ensure valid output indices
    if (out_y >= out_height || out_x >= out_width) return;
    
    // Group handling
    int group_id = out_ch * groups / out_channels;
    int in_ch_start = group_id * in_channels / groups;
    int in_ch_end = (group_id + 1) * in_channels / groups;
    
    float sum = 0.0f;
    
    // Perform convolution
    for (int in_ch = in_ch_start; in_ch < in_ch_end; ++in_ch) {
        for (int ky = 0; ky < kernel_size; ++ky) {
            for (int kx = 0; kx < kernel_size; ++kx) {
                int in_y = out_y * stride - padding + ky * dilation;
                int in_x = out_x * stride - padding + kx * dilation;
                
                if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
                    int input_idx = batch_idx * in_channels * in_height * in_width +
                                    in_ch * in_height * in_width +
                                    in_y * in_width + in_x;
                    int weight_idx = out_ch * (in_channels / groups) * kernel_size * kernel_size +
                                     (in_ch - in_ch_start) * kernel_size * kernel_size +
                                     ky * kernel_size + kx;
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    // Add bias
    sum += bias[out_ch];
    
    // Apply HardSwish: x * min(max(x + 3, 0), 6) / 6
    float hardswish_result = hardswish_impl(sum);
    
    // Apply ReLU
    float relu_result = fmaxf(hardswish_result, 0.0f);
    
    // Write output
    int output_idx = batch_idx * out_channels * out_height * out_width +
                     out_ch * out_height * out_width +
                     out_y * out_width + out_x;
    output[output_idx] = relu_result;
}

void fused_conv_hardswish_relu(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int stride,
    int padding,
    int dilation,
    int groups
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_height = input.size(2);
    int in_width = input.size(3);
    
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);
    
    int out_height = (in_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_width = (in_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    // Configure kernel launch parameters
    dim3 block(16, 16);
    dim3 grid(out_channels, batch_size, (out_height * out_width + block.x * block.y - 1) / (block.x * block.y));
    
    fused_conv_hardswish_relu_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        in_height,
        in_width,
        out_channels,
        out_height,
        out_width,
        kernel_size,
        stride,
        padding,
        dilation,
        groups
    );
}
"""

# C++ binding code
cpp_source = r"""
#include <torch/extension.h>

void fused_conv_hardswish_relu(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int stride,
    int padding,
    int dilation,
    int groups
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_hardswish_relu", &fused_conv_hardswish_relu, "Fused Conv2d + HardSwish + ReLU");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_conv_hardswish_relu_ext',
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
    # Create output tensor with correct shape
    out_channels = conv_weight.shape[0]
    in_height, in_width = x.shape[2], x.shape[3]
    kernel_size = conv_weight.shape[2]
    
    out_height = (in_height + 2 * conv_padding - conv_dilation * (kernel_size - 1) - 1) // conv_stride + 1
    out_width = (in_width + 2 * conv_padding - conv_dilation * (kernel_size - 1) - 1) // conv_stride + 1
    
    output = torch.empty((x.shape[0], out_channels, out_height, out_width), device=x.device, dtype=x.dtype)
    
    # Call the fused CUDA kernel
    fused_ext.fused_conv_hardswish_relu(
        x, conv_weight, conv_bias, output,
        conv_stride, conv_padding, conv_dilation, conv_groups
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
