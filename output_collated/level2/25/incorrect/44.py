# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_085904/code_5.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_weight', 'conv_bias', 'conv_stride', 'conv_padding', 'conv_dilation', 'conv_groups']
REQUIRED_FLAT_STATE_NAMES = ['conv_weight', 'conv_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a convolution, applies minimum operation, Tanh, and another Tanh.
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

# Define the CUDA kernel for fused operation
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

// CUDA kernel for fused convolution + min reduction + tanh activation
__global__ void fused_conv_min_tanh_kernel(
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
    int dilation,
    int groups) {
    
    // Calculate output dimensions
    int out_height = (height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_width = (width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    // Shared memory for accumulating convolution results across channels
    extern __shared__ float shared_mem[];
    
    int tid = threadIdx.x;
    int batch_idx = blockIdx.x;
    int out_y = blockIdx.y;
    int out_x = blockIdx.z;
    
    if (batch_idx >= batch_size || out_y >= out_height || out_x >= out_width) return;
    
    // Each thread processes one output position
    int out_pos = batch_idx * out_height * out_width + out_y * out_width + out_x;
    
    // Calculate input position
    int in_y_start = out_y * stride - padding;
    int in_x_start = out_x * stride - padding;
    
    // Process each output channel
    for (int out_ch = 0; out_ch < out_channels; out_ch++) {
        float sum = 0.0f;
        
        // Apply convolution for this output channel
        for (int in_ch = 0; in_ch < in_channels; in_ch++) {
            for (int ky = 0; ky < kernel_size; ky++) {
                for (int kx = 0; kx < kernel_size; kx++) {
                    int in_y = in_y_start + ky * dilation;
                    int in_x = in_x_start + kx * dilation;
                    
                    // Check bounds
                    if (in_y >= 0 && in_y < height && in_x >= 0 && in_x < width) {
                        int input_idx = batch_idx * (in_channels * height * width) + 
                                       in_ch * (height * width) + 
                                       in_y * width + in_x;
                        int weight_idx = out_ch * (in_channels * kernel_size * kernel_size) + 
                                        in_ch * (kernel_size * kernel_size) + 
                                        ky * kernel_size + kx;
                        
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        
        // Add bias
        sum += bias[out_ch];
        
        // Store in shared memory
        if (tid < out_channels) {
            shared_mem[tid] = sum;
        }
        __syncthreads();
        
        // Perform reduction to find minimum across channels
        if (tid == 0) {
            float min_val = shared_mem[0];
            for (int i = 1; i < out_channels; i++) {
                if (shared_mem[i] < min_val) {
                    min_val = shared_mem[i];
                }
            }
            
            // Apply tanh activation
            float result = tanhf(min_val);
            
            // Store result
            output[out_pos] = result;
        }
        __syncthreads();
    }
}
"""

# C++ interface code
cpp_source = r"""
#include <torch/extension.h>

void fused_conv_min_tanh_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    int groups);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_min_tanh", &fused_conv_min_tanh_kernel, "Fused Convolution + Min + Tanh");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_conv_min_tanh_ext',
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
    batch_size, in_channels, height, width = x.shape
    out_channels, _, kernel_size, _ = conv_weight.shape
    
    # Calculate output dimensions
    out_height = (height + 2 * conv_padding - conv_dilation * (kernel_size - 1) - 1) // conv_stride + 1
    out_width = (width + 2 * conv_padding - conv_dilation * (kernel_size - 1) - 1) // conv_stride + 1
    
    # Create output tensor
    output = torch.empty(batch_size, 1, out_height, out_width, device=x.device, dtype=x.dtype)
    
    # Configure kernel launch parameters
    threads_per_block = min(256, out_channels)
    blocks_per_grid = (batch_size, out_height, out_width)
    
    # Shared memory size (in bytes)
    shared_mem_size = out_channels * 4  # 4 bytes per float
    
    # Launch kernel
    fused_ext.fused_conv_min_tanh(
        x.contiguous(), 
        conv_weight.contiguous(), 
        conv_bias.contiguous(), 
        output,
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        kernel_size,
        conv_stride,
        conv_padding,
        conv_dilation,
        conv_groups,
        block=(threads_per_block,),
        grid=blocks_per_grid,
        shared_mem=shared_mem_size
    )
    
    return output

# Constants
batch_size = 128
in_channels = 16
out_channels = 64
height = width = 256
kernel_size = 3

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]
