# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_113742/code_0.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'min_value', 'divisor']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'min_value', 'divisor']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias']


class ModelNew(nn.Module):
    """
    A model that performs a transposed 3D convolution, clamps the output to a minimum value, 
    and then divides the result by a constant.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, min_value, divisor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.min_value = min_value
        self.divisor = divisor

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
    # State for conv_transpose (nn.ConvTranspose3d)
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
    if 'min_value' in flat_state:
        state_kwargs['min_value'] = flat_state['min_value']
    else:
        state_kwargs['min_value'] = getattr(model, 'min_value')
    if 'divisor' in flat_state:
        state_kwargs['divisor'] = flat_state['divisor']
    else:
        state_kwargs['divisor'] = getattr(model, 'divisor')
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

# CUDA kernel for fused conv transpose + clamp + div operations
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

__global__ void fused_conv_transpose_clamp_div_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_depth,
    int input_height,
    int input_width,
    int output_depth,
    int output_height,
    int output_width,
    int kernel_size,
    int stride,
    int padding,
    float min_value,
    float divisor) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * output_depth * output_height * output_width;
    
    if (tid >= total_elements) return;
    
    // Calculate output indices
    int temp = tid;
    int out_w = temp % output_width;
    temp /= output_width;
    int out_h = temp % output_height;
    temp /= output_height;
    int out_d = temp % output_depth;
    temp /= output_depth;
    int out_c = temp % out_channels;
    int batch_idx = temp / out_channels;
    
    // Calculate convolution
    float sum = 0.0f;
    
    // Loop through kernel
    for (int kd = 0; kd < kernel_size; kd++) {
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                // Calculate corresponding input position
                int in_d = out_d + padding - kd * stride;
                int in_h = out_h + padding - kh * stride;
                int in_w = out_w + padding - kw * stride;
                
                // Check if in valid input range
                if (in_d >= 0 && in_d < input_depth &&
                    in_h >= 0 && in_h < input_height &&
                    in_w >= 0 && in_w < input_width) {
                    
                    // Loop through input channels (assuming groups = 1 for now)
                    for (int ic = 0; ic < in_channels; ic++) {
                        int input_idx = batch_idx * (in_channels * input_depth * input_height * input_width) +
                                       ic * (input_depth * input_height * input_width) +
                                       in_d * (input_height * input_width) +
                                       in_h * input_width +
                                       in_w;
                                       
                        int weight_idx = out_c * (in_channels * kernel_size * kernel_size * kernel_size) +
                                        ic * (kernel_size * kernel_size * kernel_size) +
                                        kd * (kernel_size * kernel_size) +
                                        kh * kernel_size +
                                        kw;
                        
                        if (input_idx < batch_size * in_channels * input_depth * input_height * input_width &&
                            weight_idx < out_channels * in_channels * kernel_size * kernel_size * kernel_size) {
                            sum += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
    }
    
    // Add bias
    sum += bias[out_c];
    
    // Clamp operation
    if (sum < min_value) {
        sum = min_value;
    }
    
    // Division operation
    sum = sum / divisor;
    
    // Store result
    output[tid] = sum;
}

void fused_conv_transpose_clamp_div_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    int kernel_size,
    int stride,
    int padding,
    float min_value,
    float divisor) {
    
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_depth = input.size(2);
    int input_height = input.size(3);
    int input_width = input.size(4);
    
    int out_channels = weight.size(0);
    int output_depth = (input_depth - 1) * stride - 2 * padding + kernel_size;
    int output_height = (input_height - 1) * stride - 2 * padding + kernel_size;
    int output_width = (input_width - 1) * stride - 2 * padding + kernel_size;
    
    int total_elements = batch_size * out_channels * output_depth * output_height * output_width;
    
    const int threads = 512;
    const int blocks = (total_elements + threads - 1) / threads;
    
    fused_conv_transpose_clamp_div_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_depth,
        input_height,
        input_width,
        output_depth,
        output_height,
        output_width,
        kernel_size,
        stride,
        padding,
        min_value,
        divisor
    );
}
"""

# C++ bindings
cpp_source = r"""
#include <torch/extension.h>

void fused_conv_transpose_clamp_div_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    int kernel_size,
    int stride,
    int padding,
    float min_value,
    float divisor);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_transpose_clamp_div", &fused_conv_transpose_clamp_div_forward, "Fused ConvTranspose3d + Clamp + Div");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_conv_transpose_clamp_div',
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
    min_value,
    divisor,
):
    # Calculate output dimensions manually to match the original behavior
    # For simplicity, assuming stride is the same for all dimensions
    kernel_size = conv_transpose_weight.shape[2]  # assuming square kernel
    stride = conv_transpose_stride[0] if isinstance(conv_transpose_stride, (list, tuple)) else conv_transpose_stride
    padding = conv_transpose_padding[0] if isinstance(conv_transpose_padding, (list, tuple)) else conv_transpose_padding
    
    # Calculate output spatial dimensions
    output_depth = (x.shape[2] - 1) * stride - 2 * padding + kernel_size
    output_height = (x.shape[3] - 1) * stride - 2 * padding + kernel_size
    output_width = (x.shape[4] - 1) * stride - 2 * padding + kernel_size
    
    # Create output tensor
    output_shape = (x.shape[0], conv_transpose_weight.shape[0], output_depth, output_height, output_width)
    output = torch.empty(output_shape, device=x.device, dtype=x.dtype)
    
    # Call fused kernel
    fused_ext.fused_conv_transpose_clamp_div(
        x, conv_transpose_weight, conv_transpose_bias, output,
        kernel_size, stride, padding, min_value, divisor
    )
    
    return output

batch_size = 16
in_channels = 64
out_channels = 128
depth, height, width = 24, 48, 48
kernel_size = 3
stride = 2
padding = 1
min_value = -1.0
divisor = 2.0

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, min_value, divisor]

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width, device='cuda')]
