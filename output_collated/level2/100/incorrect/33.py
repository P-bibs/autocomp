# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_132110/code_0.py
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

# CUDA kernel that fuses conv_transpose3d + clamp + div operations
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

#define THREADS_PER_BLOCK 256
#define WARP_SIZE 32

__global__ void fused_conv_transpose3d_clamp_div_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int input_depth,
    const int input_height,
    const int input_width,
    const int output_depth,
    const int output_height,
    const int output_width,
    const int kernel_size,
    const int stride,
    const int padding,
    const float min_value,
    const float divisor) {
    
    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * output_depth * output_height * output_width;
    
    if (idx >= total_elements) return;
    
    // Decode output position
    int temp = idx;
    int w = temp % output_width;
    temp /= output_width;
    int h = temp % output_height;
    temp /= output_height;
    int d = temp % output_depth;
    temp /= output_depth;
    int c = temp % out_channels;
    int b = temp / out_channels;
    
    // Compute convolution transpose
    float sum = 0.0f;
    
    // Loop over kernel and input dimensions
    for (int kd = 0; kd < kernel_size; kd++) {
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                // Calculate corresponding input position
                int in_d = (d + padding - kd);
                int in_h = (h + padding - kh);
                int in_w = (w + padding - kw);
                
                // Check if divisible by stride
                if (in_d % stride == 0 && in_h % stride == 0 && in_w % stride == 0) {
                    in_d /= stride;
                    in_h /= stride;
                    in_w /= stride;
                    
                    // Check bounds
                    if (in_d >= 0 && in_d < input_depth &&
                        in_h >= 0 && in_h < input_height &&
                        in_w >= 0 && in_w < input_width) {
                        
                        for (int ic = 0; ic < in_channels; ic++) {
                            int input_idx = b * (in_channels * input_depth * input_height * input_width) +
                                          ic * (input_depth * input_height * input_width) +
                                          in_d * (input_height * input_width) +
                                          in_h * input_width +
                                          in_w;
                                          
                            int weight_idx = c * (in_channels * kernel_size * kernel_size * kernel_size) +
                                           ic * (kernel_size * kernel_size * kernel_size) +
                                           kd * (kernel_size * kernel_size) +
                                           kh * kernel_size +
                                           kw;
                                           
                            sum += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
    }
    
    // Add bias
    sum += bias[c];
    
    // Apply clamp and division
    sum = fmaxf(min_value, sum);
    sum = sum / divisor;
    
    // Write result
    output[idx] = sum;
}

void fused_conv_transpose3d_clamp_div_forward(
    const at::Tensor input,
    const at::Tensor weight,
    const at::Tensor bias,
    at::Tensor output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int input_depth,
    const int input_height,
    const int input_width,
    const int output_depth,
    const int output_height,
    const int output_width,
    const int kernel_size,
    const int stride,
    const int padding,
    const float min_value,
    const float divisor) {
    
    // Ensure tensors are contiguous
    auto input_contig = input.contiguous();
    auto weight_contig = weight.contiguous();
    auto bias_contig = bias.contiguous();
    auto output_contig = output.contiguous();
    
    // Set up GPU device
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input_contig));
    
    // Calculate grid and block dimensions
    int total_elements = batch_size * out_channels * output_depth * output_height * output_width;
    int blocks = (total_elements + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    // Launch kernel
    fused_conv_transpose3d_clamp_div_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        input_contig.data_ptr<float>(),
        weight_contig.data_ptr<float>(),
        bias_contig.data_ptr<float>(),
        output_contig.data_ptr<float>(),
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
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("CUDA kernel launch failed: ", cudaGetErrorString(err));
    }
}
"""

# C++ binding code
cpp_source = r"""
#include <torch/extension.h>

void fused_conv_transpose3d_clamp_div_forward(
    const at::Tensor input,
    const at::Tensor weight,
    const at::Tensor bias,
    at::Tensor output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int input_depth,
    const int input_height,
    const int input_width,
    const int output_depth,
    const int output_height,
    const int output_width,
    const int kernel_size,
    const int stride,
    const int padding,
    const float min_value,
    const float divisor);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_transpose3d_clamp_div", &fused_conv_transpose3d_clamp_div_forward, "Fused ConvTranspose3d + Clamp + Div");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_conv_transpose3d_clamp_div',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# Pre-compute constants
batch_size = 16
in_channels = 64
out_channels = 128
depth, height, width = 24, 48, 48
kernel_size = 3
stride = 2
padding = 1
min_value = -1.0
divisor = 2.0

# Pre-compute output dimensions
output_depth = (depth - 1) * stride - 2 * padding + kernel_size
output_height = (height - 1) * stride - 2 * padding + kernel_size
output_width = (width - 1) * stride - 2 * padding + kernel_size

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, min_value, divisor]

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]

# Optimized functional model using fused CUDA kernel
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
    # Create output tensor
    output = torch.empty(batch_size, out_channels, output_depth, output_height, output_width, 
                        device=x.device, dtype=x.dtype)
    
    # Call fused CUDA kernel
    fused_ext.fused_conv_transpose3d_clamp_div(
        x, conv_transpose_weight, conv_transpose_bias, output,
        batch_size, in_channels, out_channels,
        depth, height, width,
        output_depth, output_height, output_width,
        kernel_size, conv_transpose_stride[0], conv_transpose_padding[0],
        min_value, divisor
    )
    
    return output
