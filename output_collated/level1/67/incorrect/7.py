# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_155649/code_0.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'dilation', 'groups', 'bias']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv1d_weight', 'conv1d_bias', 'conv1d_stride', 'conv1d_padding', 'conv1d_dilation', 'conv1d_groups']
REQUIRED_FLAT_STATE_NAMES = ['conv1d_weight', 'conv1d_bias']


class ModelNew(nn.Module):
    """
    Performs a standard 1D convolution operation.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int=1, padding: int=0, dilation: int=1, groups: int=1, bias: bool=False):
        super(ModelNew, self).__init__()
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

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
    # State for conv1d (nn.Conv1d)
    if 'conv1d_weight' in flat_state:
        state_kwargs['conv1d_weight'] = flat_state['conv1d_weight']
    else:
        state_kwargs['conv1d_weight'] = getattr(model.conv1d, 'weight', None)
    if 'conv1d_bias' in flat_state:
        state_kwargs['conv1d_bias'] = flat_state['conv1d_bias']
    else:
        state_kwargs['conv1d_bias'] = getattr(model.conv1d, 'bias', None)
    state_kwargs['conv1d_stride'] = model.conv1d.stride
    state_kwargs['conv1d_padding'] = model.conv1d.padding
    state_kwargs['conv1d_dilation'] = model.conv1d.dilation
    state_kwargs['conv1d_groups'] = model.conv1d.groups
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

# Custom CUDA kernel for optimized 1D convolution
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define THREADS_PER_BLOCK 256
#define WARP_SIZE 32

__global__ void conv1d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_length,
    int output_length,
    int kernel_size,
    int stride,
    int padding,
    int dilation) {
    
    // Shared memory for weights and partial sums
    extern __shared__ float shared_mem[];
    float* shared_weights = shared_mem;
    float* partial_sums = shared_mem + blockDim.x * kernel_size;
    
    int tid = threadIdx.x;
    int batch_idx = blockIdx.z;
    int out_ch = blockIdx.y;
    int out_pos_base = blockIdx.x * blockDim.x;
    
    if (batch_idx >= batch_size || out_ch >= out_channels) return;
    
    // Load weights into shared memory
    for (int i = tid; i < in_channels * kernel_size; i += blockDim.x) {
        int w_idx = out_ch * in_channels * kernel_size + i;
        shared_weights[i] = weight[w_idx];
    }
    __syncthreads();
    
    // Process multiple output positions per thread block
    for (int out_pos_offset = 0; out_pos_offset < blockDim.x; out_pos_offset++) {
        int out_pos = out_pos_base + out_pos_offset;
        if (out_pos >= output_length) break;
        
        float sum = 0.0f;
        
        // Convolution computation
        for (int in_ch = 0; in_ch < in_channels; in_ch++) {
            for (int k = 0; k < kernel_size; k++) {
                int input_pos = out_pos * stride - padding + k * dilation;
                
                if (input_pos >= 0 && input_pos < input_length) {
                    int input_idx = ((batch_idx * in_channels + in_ch) * input_length) + input_pos;
                    int weight_idx = in_ch * kernel_size + k;
                    sum += input[input_idx] * shared_weights[weight_idx];
                }
            }
        }
        
        // Add bias
        if (tid == 0) {
            sum += bias[out_ch];
        }
        
        // Write output
        int output_idx = ((batch_idx * out_channels + out_ch) * output_length) + out_pos;
        output[output_idx] = sum;
    }
}

// Optimized version with better memory coalescing
__global__ void conv1d_optimized_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_length,
    int output_length,
    int kernel_size,
    int stride,
    int padding,
    int dilation) {
    
    int tid = threadIdx.x;
    int batch_idx = blockIdx.z;
    int out_ch = blockIdx.y;
    int out_pos = blockIdx.x * blockDim.x + tid;
    
    if (batch_idx >= batch_size || out_ch >= out_channels) return;
    
    // Shared memory for weights
    extern __shared__ float shared_weights[];
    
    // Load weights into shared memory - coalesced access
    for (int i = tid; i < in_channels * kernel_size; i += blockDim.x) {
        int w_idx = out_ch * in_channels * kernel_size + i;
        shared_weights[i] = weight[w_idx];
    }
    __syncthreads();
    
    // Process one output position per thread
    if (out_pos < output_length) {
        float sum = 0.0f;
        
        // Convolution computation
        for (int in_ch = 0; in_ch < in_channels; in_ch++) {
            for (int k = 0; k < kernel_size; k++) {
                int input_pos = out_pos * stride - padding + k * dilation;
                
                if (input_pos >= 0 && input_pos < input_length) {
                    int input_idx = ((batch_idx * in_channels + in_ch) * input_length) + input_pos;
                    int weight_idx = in_ch * kernel_size + k;
                    sum += input[input_idx] * shared_weights[weight_idx];
                }
            }
        }
        
        // Add bias
        sum += bias[out_ch];
        
        // Write output
        int output_idx = ((batch_idx * out_channels + out_ch) * output_length) + out_pos;
        output[output_idx] = sum;
    }
}

void launch_conv1d(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    int stride,
    int padding,
    int dilation) {
    
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_length = input.size(2);
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);
    
    int output_length = (input_length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    // Kernel launch configuration
    dim3 grid((output_length + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, out_channels, batch_size);
    dim3 block(THREADS_PER_BLOCK);
    
    // Shared memory size for weights
    int shared_mem_size = in_channels * kernel_size * sizeof(float);
    
    conv1d_optimized_kernel<<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_length,
        output_length,
        kernel_size,
        stride,
        padding,
        dilation
    );
}
"""

# C++ interface/bindings
cpp_source = r"""
#include <torch/extension.h>

void launch_conv1d(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    int stride,
    int padding,
    int dilation);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv1d_cuda", &launch_conv1d, "1D convolution CUDA kernel");
}
"""

# Compile the extension
conv_ext = load_inline(
    name='conv1d_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math', '-arch=compute_75', '-code=sm_75'],
    with_cuda=True
)

def functional_model(
    x,
    *,
    conv1d_weight,
    conv1d_bias,
    conv1d_stride,
    conv1d_padding,
    conv1d_dilation,
    conv1d_groups,
):
    # Calculate output dimensions
    batch_size, in_channels, input_length = x.shape
    out_channels, _, kernel_size = conv1d_weight.shape
    
    # Calculate output length
    output_length = (input_length + 2 * conv1d_padding - conv1d_dilation * (kernel_size - 1) - 1) // conv1d_stride + 1
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, output_length, device=x.device, dtype=x.dtype)
    
    # Call custom CUDA kernel
    conv_ext.conv1d_cuda(
        x, conv1d_weight, conv1d_bias, output,
        conv1d_stride, conv1d_padding, conv1d_dilation
    )
    
    return output

batch_size = 32
in_channels = 64
out_channels = 128
kernel_size = 3
length = 131072

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]  # Provide in_channels, out_channels, kernel_size for initialization

def get_inputs():
    x = torch.rand(batch_size, in_channels, length, device='cuda')
    return [x]
