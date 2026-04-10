# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_071708/code_0.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'bias']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv1d_weight', 'conv1d_bias', 'conv1d_stride', 'conv1d_padding', 'conv1d_dilation', 'conv1d_groups']
REQUIRED_FLAT_STATE_NAMES = ['conv1d_weight', 'conv1d_bias']


class ModelNew(nn.Module):
    """
    Performs a pointwise 2D convolution operation.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """

    def __init__(self, in_channels: int, out_channels: int, bias: bool=False):
        super(ModelNew, self).__init__()
        self.conv1d = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)

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
    # State for conv1d (nn.Conv2d)
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

# Define the CUDA kernel for a fused convolution operation
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

template <typename scalar_t>
__global__ void conv1d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_width,
    int output_width,
    int kernel_size,
    int stride,
    int padding,
    int dilation) {
    
    // Shared memory for weight data
    extern __shared__ char shared_mem[];
    scalar_t* shared_weight = reinterpret_cast<scalar_t*>(shared_mem);
    
    int tid = threadIdx.x;
    int batch_idx = blockIdx.x;
    int out_ch = blockIdx.y;
    int out_x = blockIdx.z * blockDim.x + threadIdx.x;
    
    if (batch_idx >= batch_size || out_ch >= out_channels) return;
    
    // Load bias
    scalar_t bias_val = (bias != nullptr) ? bias[out_ch] : 0;
    
    // Load weight data to shared memory
    for (int i = tid; i < in_channels * kernel_size; i += blockDim.x) {
        shared_weight[i] = weight[out_ch * in_channels * kernel_size + i];
    }
    __syncthreads();
    
    // Process output position
    if (out_x < output_width) {
        scalar_t sum = bias_val;
        
        // Convolution computation
        for (int in_ch = 0; in_ch < in_channels; ++in_ch) {
            for (int k = 0; k < kernel_size; ++k) {
                int input_x = out_x * stride + k * dilation - padding;
                scalar_t input_val = 0;
                if (input_x >= 0 && input_x < input_width) {
                    input_val = input[((batch_idx * in_channels + in_ch) * input_width) + input_x];
                }
                sum += input_val * shared_weight[(in_ch * kernel_size) + k];
            }
        }
        
        output[((batch_idx * out_channels + out_ch) * output_width) + out_x] = sum;
    }
}

void conv1d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int stride,
    int padding,
    int dilation) {
    
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto input_width = input.size(2);
    auto out_channels = weight.size(0);
    auto kernel_size = weight.size(2);
    auto output_width = (input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    // Launch configuration
    const int threads_per_block = 256;
    const int blocks_x = batch_size;
    const int blocks_y = out_channels;
    const int blocks_z = (output_width + threads_per_block - 1) / threads_per_block;
    const dim3 blocks(blocks_x, blocks_y, blocks_z);
    const dim3 threads(threads_per_block);
    
    // Calculate shared memory size for weights
    const int shared_weight_size = in_channels * kernel_size;
    const int shared_mem_size = shared_weight_size * sizeof(float);
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv1d_forward", ([&] {
        conv1d_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.defined() ? bias.data_ptr<scalar_t>() : nullptr,
            output.data_ptr<scalar_t>(),
            batch_size,
            in_channels,
            out_channels,
            input_width,
            output_width,
            kernel_size,
            stride,
            padding,
            dilation
        );
    }));
}
"""

# C++ interface
cpp_source = r"""
#include <torch/extension.h>

void conv1d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int stride,
    int padding,
    int dilation);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv1d_forward", &conv1d_forward, "1D Convolution forward pass");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_conv1d',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# Global variables to store convolution parameters
conv_params = {
    'weight': None,
    'bias': None,
    'stride': 1,
    'padding': 0,
    'dilation': 1,
    'groups': 1
}

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
    # Store parameters for use
    global conv_params
    conv_params['weight'] = conv1d_weight
    conv_params['bias'] = conv1d_bias
    conv_params['stride'] = conv1d_stride[0] if isinstance(conv1d_stride, (list, tuple)) else conv1d_stride
    conv_params['padding'] = conv1d_padding[0] if isinstance(conv1d_padding, (list, tuple)) else conv1d_padding
    conv_params['dilation'] = conv1d_dilation[0] if isinstance(conv1d_dilation, (list, tuple)) else conv1d_dilation
    conv_params['groups'] = conv1d_groups
    
    # Validate groups (only supporting groups=1 for this implementation)
    if conv1d_groups != 1:
        raise ValueError("Only groups=1 is supported in this implementation")
    
    # Calculate output dimensions
    batch_size, in_channels, height, width = x.shape
    out_channels = conv1d_weight.shape[0]
    kernel_size = conv1d_weight.shape[2]
    
    # For 1D convolution, assuming height=1 or processing along width
    output_width = (width + 2 * conv_params['padding'] - conv_params['dilation'] * (kernel_size - 1) - 1) // conv_params['stride'] + 1
    output = torch.zeros(batch_size, out_channels, height, output_width, device=x.device, dtype=x.dtype)
    
    # Reshape tensors for 1D convolution (assuming height=1, working on width dimension)
    if height != 1:
        # If height > 1, we process each row independently as a separate 1D conv
        for h in range(height):
            x_slice = x[:, :, h, :].contiguous()
            output_slice = output[:, :, h, :].contiguous()
            
            fused_ext.conv1d_forward(
                x_slice, 
                conv_params['weight'], 
                conv_params['bias'] if conv_params['bias'] is not None else torch.empty(0, device=x.device), 
                output_slice,
                conv_params['stride'],
                conv_params['padding'],
                conv_params['dilation']
            )
    else:
        # Standard case where we have (batch, channels, 1, width)
        x_flat = x.view(batch_size, in_channels, width)
        output_flat = output.view(batch_size, out_channels, output_width)
        
        # Call our custom CUDA kernel
        fused_ext.conv1d_forward(
            x_flat, 
            conv_params['weight'], 
            conv_params['bias'] if conv_params['bias'] is not None else torch.empty(0, device=x.device), 
            output_flat,
            conv_params['stride'],
            conv_params['padding'],
            conv_params['dilation']
        )
    
    return output

# Test parameters
batch_size = 16
in_channels = 64
out_channels = 128
width = 1024
height = 1

def get_init_inputs():
    return [in_channels, out_channels]

def get_inputs():
    x = torch.rand(batch_size, in_channels, height, width, device='cuda')
    return [x]
