# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_122448/code_2.py
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

# Custom CUDA kernel for fused transposed convolution + clamp + division
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

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
    const int kernel_size,
    const int stride,
    const int padding,
    const float min_value,
    const float divisor,
    const int output_depth,
    const int output_height,
    const int output_width
) {
    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * output_depth * output_height * output_width;
    
    if (idx >= total_elements) return;
    
    // Decode output indices
    int temp = idx;
    int w_out = temp % output_width;
    temp /= output_width;
    int h_out = temp % output_height;
    temp /= output_height;
    int d_out = temp % output_depth;
    temp /= output_depth;
    int c_out = temp % out_channels;
    int n = temp / out_channels;
    
    // Calculate input region that contributes to this output element
    int d_start = (d_out + padding - kernel_size + 1 + stride - 1) / stride;  // Ceiling division
    int d_end = min((d_out + padding) / stride + 1, input_depth);
    d_start = max(d_start, 0);
    
    int h_start = (h_out + padding - kernel_size + 1 + stride - 1) / stride;
    int h_end = min((h_out + padding) / stride + 1, input_height);
    h_start = max(h_start, 0);
    
    int w_start = (w_out + padding - kernel_size + 1 + stride - 1) / stride;
    int w_end = min((w_out + padding) / stride + 1, input_width);
    w_start = max(w_start, 0);
    
    float sum = 0.0f;
    
    // Perform convolution computation
    for (int kd = 0; kd < kernel_size; ++kd) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int d_in = d_out + padding - kd;
                int h_in = h_out + padding - kh;
                int w_in = w_out + padding - kw;
                
                if (d_in % stride == 0 && h_in % stride == 0 && w_in % stride == 0) {
                    d_in /= stride;
                    h_in /= stride;
                    w_in /= stride;
                    
                    if (d_in >= 0 && d_in < input_depth &&
                        h_in >= 0 && h_in < input_height &&
                        w_in >= 0 && w_in < input_width) {
                        
                        for (int c_in = 0; c_in < in_channels; ++c_in) {
                            int input_idx = n * (in_channels * input_depth * input_height * input_width) +
                                          c_in * (input_depth * input_height * input_width) +
                                          d_in * (input_height * input_width) +
                                          h_in * input_width +
                                          w_in;
                                          
                            int weight_idx = c_out * (in_channels * kernel_size * kernel_size * kernel_size) +
                                           c_in * (kernel_size * kernel_size * kernel_size) +
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
    sum += bias[c_out];
    
    // Apply clamp and division
    sum = fmaxf(min_value, sum);
    sum /= divisor;
    
    // Write result
    output[idx] = sum;
}

void fused_conv_transpose3d_clamp_div_forward(
    const torch::Tensor input,
    const torch::Tensor weight,
    const torch::Tensor bias,
    torch::Tensor output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int input_depth,
    const int input_height,
    const int input_width,
    const int kernel_size,
    const int stride,
    const int padding,
    const float min_value,
    const float divisor,
    const int output_depth,
    const int output_height,
    const int output_width
) {
    // Ensure tensors are contiguous
    auto input_contig = input.contiguous();
    auto weight_contig = weight.contiguous();
    auto bias_contig = bias.contiguous();
    auto output_contig = output.contiguous();
    
    // Set up CUDA context
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    
    // Calculate grid and block dimensions
    int total_elements = batch_size * out_channels * output_depth * output_height * output_width;
    int threads_per_block = 512;
    int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    // Launch kernel
    fused_conv_transpose3d_clamp_div_kernel<<<blocks, threads_per_block>>>(
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
        kernel_size,
        stride,
        padding,
        min_value,
        divisor,
        output_depth,
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

# C++ interface/bindings
cpp_source = r"""
#include <torch/extension.h>

void fused_conv_transpose3d_clamp_div_forward(
    const torch::Tensor input,
    const torch::Tensor weight,
    const torch::Tensor bias,
    torch::Tensor output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int input_depth,
    const int input_height,
    const int input_width,
    const int kernel_size,
    const int stride,
    const int padding,
    const float min_value,
    const float divisor,
    const int output_depth,
    const int output_height,
    const int output_width
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_transpose3d_clamp_div_forward", 
          &fused_conv_transpose3d_clamp_div_forward, 
          "Fused 3D transposed convolution with clamp and division");
}
"""

# Compile the extension
fused_conv_ext = load_inline(
    name='fused_conv_op',
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
    # Calculate output dimensions
    batch_size, in_channels, input_depth, input_height, input_width = x.shape
    out_channels = conv_transpose_weight.shape[0]
    kernel_size = conv_transpose_weight.shape[2]
    
    # Calculate output dimensions for transposed convolution
    output_depth = (input_depth - 1) * conv_transpose_stride[0] - 2 * conv_transpose_padding[0] + kernel_size + conv_transpose_output_padding[0]
    output_height = (input_height - 1) * conv_transpose_stride[1] - 2 * conv_transpose_padding[1] + kernel_size + conv_transpose_output_padding[1]
    output_width = (input_width - 1) * conv_transpose_stride[2] - 2 * conv_transpose_padding[2] + kernel_size + conv_transpose_output_padding[2]
    
    # Create output tensor
    output = torch.empty((batch_size, out_channels, output_depth, output_height, output_width), 
                         dtype=x.dtype, device=x.device)
    
    # Call fused kernel
    fused_conv_ext.fused_conv_transpose3d_clamp_div_forward(
        x, conv_transpose_weight, conv_transpose_bias, output,
        batch_size, in_channels, out_channels,
        input_depth, input_height, input_width,
        kernel_size,
        conv_transpose_stride[0],  # Assuming uniform stride
        conv_transpose_padding[0],  # Assuming uniform padding
        min_value, divisor,
        output_depth, output_height, output_width
    )
    
    return output

# Constants (same as original)
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
    return [torch.rand(batch_size, in_channels, depth, height, width)]
