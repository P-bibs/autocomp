# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_152911/code_2.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'add_value', 'multiply_value']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'add_value', 'multiply_value']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a transposed convolution, adds a value, takes the minimum, applies GELU, and multiplies by a value.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, add_value, multiply_value):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.add_value = add_value
        self.multiply_value = multiply_value

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
    # State for conv_transpose (nn.ConvTranspose2d)
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
    if 'add_value' in flat_state:
        state_kwargs['add_value'] = flat_state['add_value']
    else:
        state_kwargs['add_value'] = getattr(model, 'add_value')
    if 'multiply_value' in flat_state:
        state_kwargs['multiply_value'] = flat_state['multiply_value']
    else:
        state_kwargs['multiply_value'] = getattr(model, 'multiply_value')
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
import math
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernel for fused operations
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__device__ float gelu_approx(float x) {
    // Fast GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const float sqrt_two_over_pi = 0.7978845608028654f;  // sqrt(2/pi)
    const float coeff = 0.044715f;
    float x_cubed = x * x * x;
    float inner = sqrt_two_over_pi * (x + coeff * x_cubed);
    return 0.5f * x * (1.0f + tanhf(inner));
}

__global__ void fused_op_kernel(
    const float* input,
    float* output,
    const float add_value,
    const float multiply_value,
    const int num_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        float val = input[idx];
        val += add_value;                         // Add
        val = fminf(val, 0.0f);                  // Min with 0 (clipping)
        val = gelu_approx(val);                  // GELU activation
        val *= multiply_value;                   // Multiply
        output[idx] = val;
    }
}

void fused_op_forward(
    const torch::Tensor input,
    torch::Tensor output,
    const float add_value,
    const float multiply_value
) {
    const int num_elements = input.numel();
    const int threads_per_block = 256;
    const int blocks = (num_elements + threads_per_block - 1) / threads_per_block;
    
    fused_op_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        add_value,
        multiply_value,
        num_elements
    );
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR(cudaGetErrorString(err));
    }
}
"""

# C++ binding code
cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(
    const torch::Tensor input,
    torch::Tensor output,
    const float add_value,
    const float multiply_value
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused element-wise operations");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_op_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# Custom CUDA kernel for transposed convolution
conv_transpose_cuda = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

__global__ void conv_transpose2d_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int in_height,
    const int in_width,
    const int out_height,
    const int out_width,
    const int kernel_size,
    const int stride,
    const int padding,
    const int output_padding,
    const int groups,
    const int dilation
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = batch_size * out_channels * out_height * out_width;
    
    if (out_idx >= total_outputs) return;
    
    int tmp = out_idx;
    const int w_out = tmp % out_width;
    tmp /= out_width;
    const int h_out = tmp % out_height;
    tmp /= out_height;
    const int c_out = tmp % out_channels;
    const int n = tmp / out_channels;
    
    float sum = 0.0f;
    
    // Calculate input ranges
    const int kernel_radius = (kernel_size - 1) / 2;
    const int h_in_start = max(0, (h_out + padding - kernel_size + 1 + stride - 1) / stride);
    const int h_in_end = min(in_height, (h_out + padding + stride - 1) / stride + 1);
    const int w_in_start = max(0, (w_out + padding - kernel_size + 1 + stride - 1) / stride);
    const int w_in_end = min(in_width, (w_out + padding + stride - 1) / stride + 1);
    
    // Process valid input positions
    for (int h_in = h_in_start; h_in < h_in_end; ++h_in) {
        for (int w_in = w_in_start; w_in < w_in_end; ++w_in) {
            // Calculate kernel position
            int kh = h_out - h_in * stride + padding;
            int kw = w_out - w_in * stride + padding;
            
            // Check if kernel position is valid
            if (kh >= 0 && kh < kernel_size && kw >= 0 && kw < kernel_size) {
                int group_id = c_out / (out_channels / groups);
                int c_in_group = c_out % (out_channels / groups);
                int c_in = group_id * (in_channels / groups) + c_in_group;
                
                // Input index
                int input_idx = ((n * in_channels + c_in) * in_height + h_in) * in_width + w_in;
                
                // Weight index
                int weight_idx = ((c_out * in_channels / groups + c_in_group) * kernel_size + kh) * kernel_size + kw;
                
                sum += input[input_idx] * weight[weight_idx];
            }
        }
    }
    
    // Add bias
    sum += bias[c_out];
    
    output[out_idx] = sum;
}

void conv_transpose2d_cuda(
    const torch::Tensor input,
    const torch::Tensor weight,
    const torch::Tensor bias,
    torch::Tensor output,
    const int stride,
    const int padding,
    const int output_padding,
    const int groups,
    const int dilation
) {
    const auto input_sizes = input.sizes();
    const auto weight_sizes = weight.sizes();
    const auto output_sizes = output.sizes();
    
    const int batch_size = input_sizes[0];
    const int in_channels = input_sizes[1];
    const int in_height = input_sizes[2];
    const int in_width = input_sizes[3];
    
    const int out_channels = output_sizes[1];
    const int out_height = output_sizes[2];
    const int out_width = output_sizes[3];
    
    const int kernel_size = weight_sizes[2];
    
    const int total_threads = batch_size * out_channels * out_height * out_width;
    const int threads_per_block = 256;
    const int blocks = (total_threads + threads_per_block - 1) / threads_per_block;
    
    at::cuda::CUDAGuard device_guard(input.device());
    
    conv_transpose2d_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        in_height,
        in_width,
        out_height,
        out_width,
        kernel_size,
        stride,
        padding,
        output_padding,
        groups,
        dilation
    );
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR(cudaGetErrorString(err));
    }
}
"""

# C++ binding for conv transpose
conv_transpose_cpp = r"""
#include <torch/extension.h>

void conv_transpose2d_cuda(
    const torch::Tensor input,
    const torch::Tensor weight,
    const torch::Tensor bias,
    torch::Tensor output,
    const int stride,
    const int padding,
    const int output_padding,
    const int groups,
    const int dilation
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_transpose2d", &conv_transpose2d_cuda, "Custom Conv Transpose 2D");
}
"""

# Compile the conv transpose extension
conv_transpose_ext = load_inline(
    name='conv_transpose_ext',
    cpp_sources=conv_transpose_cpp,
    cuda_sources=conv_transpose_cuda,
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
    add_value,
    multiply_value,
):
    # Perform transposed convolution using our custom CUDA kernel
    out = torch.empty(
        (x.size(0), conv_transpose_weight.size(1), 
         (x.size(2) - 1) * conv_transpose_stride + conv_transpose_weight.size(2) - 2 * conv_transpose_padding + conv_transpose_output_padding,
         (x.size(3) - 1) * conv_transpose_stride + conv_transpose_weight.size(3) - 2 * conv_transpose_padding + conv_transpose_output_padding),
        dtype=x.dtype,
        device=x.device
    )
    
    conv_transpose_ext.conv_transpose2d(
        x, conv_transpose_weight, conv_transpose_bias, out,
        conv_transpose_stride, conv_transpose_padding, conv_transpose_output_padding,
        conv_transpose_groups, conv_transpose_dilation
    )
    
    # Apply fused operations using our custom CUDA kernel
    result = torch.empty_like(out)
    fused_ext.fused_op(out, result, add_value, multiply_value)
    
    return result

batch_size = 128
in_channels = 64
out_channels = 128
height, width = 64, 64
kernel_size = 4
stride = 2
add_value = 0.5
multiply_value = 2.0

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, add_value, multiply_value]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]
