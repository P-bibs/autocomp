# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_114641/code_5.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'bias_shape', 'stride', 'padding', 'output_padding']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'bias']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a transposed convolution, subtracts a bias term, and applies tanh activation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, bias_shape, stride=2, padding=1, output_padding=1):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))

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
    if 'bias' in flat_state:
        state_kwargs['bias'] = flat_state['bias']
    else:
        state_kwargs['bias'] = getattr(model, 'bias')
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

# Custom CUDA kernel for fused bias subtraction and tanh activation
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_op_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int out_channels,
    const int height,
    const int width
) {
    const int total_elements = batch_size * out_channels * height * width;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < total_elements; i += stride) {
        // Calculate channel index for bias access
        const int channel_idx = (i / (height * width)) % out_channels;
        
        // Perform bias subtraction and tanh activation
        const float val = input[i] - bias[channel_idx];
        output[i] = tanhf(val);
    }
}

void fused_op_forward(
    const torch::Tensor input,
    const torch::Tensor bias,
    torch::Tensor output
) {
    const int batch_size = input.size(0);
    const int out_channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);
    
    const int total_elements = batch_size * out_channels * height * width;
    
    // Launch configuration
    const int threads_per_block = 256;
    const int blocks = std::min(
        (total_elements + threads_per_block - 1) / threads_per_block,
        65535
    );
    
    fused_op_forward_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        out_channels,
        height,
        width
    );
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("CUDA kernel launch failed: ", cudaGetErrorString(err));
    }
}
"""

# C++ interface bindings
cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(
    const torch::Tensor input,
    const torch::Tensor bias,
    torch::Tensor output
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused bias subtraction and tanh activation");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# Custom CUDA kernel for transposed convolution
conv_transpose_cuda = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void conv_transpose2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int input_height,
    const int input_width,
    const int kernel_size,
    const int stride,
    const int padding,
    const int output_padding,
    const int groups,
    const int dilation
) {
    const int output_height = (input_height - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1 + output_padding;
    const int output_width = (input_width - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1 + output_padding;
    
    const int group_size = in_channels / groups;
    const int out_group_size = out_channels / groups;
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = batch_size * out_channels * output_height * output_width;
    
    for (int idx = tid; idx < total_threads; idx += blockDim.x * gridDim.x) {
        const int w = idx % output_width;
        const int h = (idx / output_width) % output_height;
        const int c_out = (idx / (output_width * output_height)) % out_channels;
        const int n = idx / (output_width * output_height * out_channels);
        
        const int group = c_out / out_group_size;
        
        float sum = 0.0f;
        
        // Calculate corresponding input position
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                // Calculate input position that would contribute to this output position
                const int ih = h + padding - kh * dilation;
                const int iw = w + padding - kw * dilation;
                
                if (ih % stride == 0 && iw % stride == 0) {
                    const int ih_s = ih / stride;
                    const int iw_s = iw / stride;
                    
                    if (ih_s >= 0 && ih_s < input_height && iw_s >= 0 && iw_s < input_width) {
                        for (int c_in_group = 0; c_in_group < group_size; ++c_in_group) {
                            const int c_in = group * group_size + c_in_group;
                            
                            const int weight_idx = c_in * (out_group_size * kernel_size * kernel_size) +
                                                  (c_out % out_group_size) * (kernel_size * kernel_size) +
                                                  (kernel_size - 1 - kh) * kernel_size +
                                                  (kernel_size - 1 - kw);
                                                  
                            const int input_idx = n * (in_channels * input_height * input_width) +
                                                 c_in * (input_height * input_width) +
                                                 ih_s * input_width +
                                                 iw_s;
                                                 
                            sum += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
        
        output[idx] = sum + bias[c_out];
    }
}

void conv_transpose2d_forward(
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
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int input_height = input.size(2);
    const int input_width = input.size(3);
    const int out_channels = weight.size(1);
    const int kernel_size = weight.size(2);
    
    const int output_height = (input_height - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1 + output_padding;
    const int output_width = (input_width - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1 + output_padding;
    
    const int total_elements = batch_size * out_channels * output_height * output_width;
    
    const int threads_per_block = 256;
    const int blocks = std::min(
        (total_elements + threads_per_block - 1) / threads_per_block,
        65535
    );
    
    conv_transpose2d_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_height,
        input_width,
        kernel_size,
        stride,
        padding,
        output_padding,
        groups,
        dilation
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("CUDA kernel launch failed: ", cudaGetErrorString(err));
    }
}
"""

conv_transpose_cpp = r"""
#include <torch/extension.h>

void conv_transpose2d_forward(
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
    m.def("conv_transpose2d", &conv_transpose2d_forward, "Custom ConvTranspose2D implementation");
}
"""

# Compile the conv transpose extension
conv_transpose_ext = load_inline(
    name='conv_transpose2d_custom',
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
    bias,
):
    # Perform transposed convolution using custom CUDA kernel
    out_channels = conv_transpose_weight.size(1)
    kernel_size = conv_transpose_weight.size(2)
    
    # Calculate output dimensions
    input_height, input_width = x.size(2), x.size(3)
    output_height = (input_height - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_dilation * (kernel_size - 1) + 1 + conv_transpose_output_padding
    output_width = (input_width - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_dilation * (kernel_size - 1) + 1 + conv_transpose_output_padding
    
    # Create output tensor for conv transpose
    conv_output = torch.empty(
        x.size(0), out_channels, output_height, output_width,
        dtype=x.dtype, device=x.device
    )
    
    # Run custom conv transpose
    conv_transpose_ext.conv_transpose2d(
        x, conv_transpose_weight, conv_transpose_bias, conv_output,
        conv_transpose_stride, conv_transpose_padding, conv_transpose_output_padding,
        conv_transpose_groups, conv_transpose_dilation
    )
    
    # Apply fused bias subtraction and tanh activation
    final_output = torch.empty_like(conv_output)
    fused_ext.fused_op(conv_output, bias, final_output)
    
    return final_output

batch_size = 32
in_channels  = 64  
out_channels = 64  
height = width = 256 
kernel_size = 4
bias_shape = (out_channels, 1, 1)

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]
