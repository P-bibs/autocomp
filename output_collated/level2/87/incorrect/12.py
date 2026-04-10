# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_141459/code_0.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'subtract_value_1', 'subtract_value_2']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_weight', 'conv_bias', 'conv_stride', 'conv_padding', 'conv_dilation', 'conv_groups', 'subtract_value_1', 'subtract_value_2']
REQUIRED_FLAT_STATE_NAMES = ['conv_weight', 'conv_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a convolution, subtracts two values, applies Mish activation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value_1 = subtract_value_1
        self.subtract_value_2 = subtract_value_2

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
    if 'subtract_value_1' in flat_state:
        state_kwargs['subtract_value_1'] = flat_state['subtract_value_1']
    else:
        state_kwargs['subtract_value_1'] = getattr(model, 'subtract_value_1')
    if 'subtract_value_2' in flat_state:
        state_kwargs['subtract_value_2'] = flat_state['subtract_value_2']
    else:
        state_kwargs['subtract_value_2'] = getattr(model, 'subtract_value_2')
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

# CUDA kernel that fuses conv2d + double subtraction + mish
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

__device__ float mish_activation(float x) {
    return x * tanhf(log1pf(expf(x)));
}

__global__ void fused_conv_subtract_mish_kernel_opt(
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
    float subtract_value_1,
    float subtract_value_2) {
    
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_ch = blockIdx.z;
    
    int height_out = (height + 2 * padding - kernel_size) / stride + 1;
    int width_out = (width + 2 * padding - kernel_size) / stride + 1;
    
    if (out_x >= width_out || out_y >= height_out || out_ch >= out_channels) return;
    
    for (int batch = 0; batch < batch_size; batch++) {
        float sum = 0.0f;
        
        // Perform convolution
        for (int in_ch = 0; in_ch < in_channels; in_ch++) {
            for (int ky = 0; ky < kernel_size; ky++) {
                for (int kx = 0; kx < kernel_size; kx++) {
                    int in_x = out_x * stride - padding + kx;
                    int in_y = out_y * stride - padding + ky;
                    
                    if (in_x >= 0 && in_x < width && in_y >= 0 && in_y < height) {
                        int input_idx = ((batch * in_channels + in_ch) * height + in_y) * width + in_x;
                        int weight_idx = ((out_ch * in_channels + in_ch) * kernel_size + ky) * kernel_size + kx;
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        
        // Add bias and subtract values in one step
        sum += bias[out_ch] - subtract_value_1 - subtract_value_2;
        
        // Apply Mish activation
        float result = sum * tanhf(log1pf(expf(sum)));
        
        // Write output
        int output_idx = ((batch * out_channels + out_ch) * height_out + out_y) * width_out + out_x;
        output[output_idx] = result;
    }
}

void fused_conv_subtract_mish_forward(
    const at::Tensor input,
    const at::Tensor weight,
    const at::Tensor bias,
    at::Tensor output,
    int kernel_size,
    int stride,
    int padding,
    float subtract_value_1,
    float subtract_value_2) {
    
    const auto batch_size = input.size(0);
    const auto in_channels = input.size(1);
    const auto height = input.size(2);
    const auto width = input.size(3);
    const auto out_channels = weight.size(0);
    
    int height_out = (height + 2 * padding - kernel_size) / stride + 1;
    int width_out = (width + 2 * padding - kernel_size) / stride + 1;
    
    const dim3 block_size(16, 16, 1);
    const dim3 grid_size(
        (width_out + block_size.x - 1) / block_size.x,
        (height_out + block_size.y - 1) / block_size.y,
        out_channels
    );
    
    // Ensure we're on the right device
    at::cuda::CUDAGuard device_guard(input.device());
    
    fused_conv_subtract_mish_kernel_opt<<<grid_size, block_size>>>(
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
        subtract_value_1,
        subtract_value_2
    );
    
    cudaDeviceSynchronize();
}
"""

# C++ interface/bindings
cpp_source = r"""
#include <torch/extension.h>

void fused_conv_subtract_mish_forward(
    const at::Tensor input,
    const at::Tensor weight,
    const at::Tensor bias,
    at::Tensor output,
    int kernel_size,
    int stride,
    int padding,
    float subtract_value_1,
    float subtract_value_2);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_subtract_mish", &fused_conv_subtract_mish_forward, "Fused Conv2D + Subtract + Mish");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_conv_subtract_mish_ext',
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
    subtract_value_1,
    subtract_value_2,
):
    # Validate that we're using the expected parameters
    assert conv_dilation == 1, "Only dilation=1 is supported in this optimization"
    assert conv_groups == 1, "Only groups=1 is supported in this optimization"
    assert isinstance(conv_stride, int) or (len(conv_stride) == 2 and conv_stride[0] == conv_stride[1]), "Only square strides supported"
    assert isinstance(conv_padding, int) or (len(conv_padding) == 2 and conv_padding[0] == conv_padding[1]), "Only square padding supported"
    
    stride_val = conv_stride if isinstance(conv_stride, int) else conv_stride[0]
    padding_val = conv_padding if isinstance(conv_padding, int) else conv_padding[0]
    
    # Create output tensor with the same shape as what conv2d would produce
    out_height = (x.shape[2] + 2 * padding_val - conv_weight.shape[2]) // stride_val + 1
    out_width = (x.shape[3] + 2 * padding_val - conv_weight.shape[3]) // stride_val + 1
    output = torch.empty(x.shape[0], conv_weight.shape[0], out_height, out_width, device=x.device, dtype=x.dtype)
    
    # Use our fused kernel
    fused_ext.fused_conv_subtract_mish(
        x,
        conv_weight,
        conv_bias,
        output,
        conv_weight.shape[2],  # kernel_size
        stride_val,
        padding_val,
        subtract_value_1,
        subtract_value_2
    )
    
    return output

batch_size = 128
in_channels = 8
out_channels = 64
height, width = 256, 256
kernel_size = 3
subtract_value_1 = 0.5
subtract_value_2 = 0.2

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width, device='cuda')]
