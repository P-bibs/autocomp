# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_083856/code_1.py
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

# CUDA kernel implementation
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ATen.h>

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
    int dilation) {
    
    // Calculate output dimensions
    int out_height = (height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_width = (width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    // Thread indices
    int batch_idx = blockIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_x = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (batch_idx >= batch_size || out_y >= out_height || out_x >= out_width) return;
    
    // Shared memory for min reduction
    extern __shared__ float shared_min[];
    float min_val = INFINITY;
    
    // Perform convolution and find min across channels
    for (int out_ch = 0; out_ch < out_channels; out_ch++) {
        float conv_result = 0.0f;
        
        // Perform convolution for this output channel
        for (int in_ch = 0; in_ch < in_channels; in_ch++) {
            for (int ky = 0; ky < kernel_size; ky++) {
                for (int kx = 0; kx < kernel_size; kx++) {
                    // Calculate input position
                    int in_y = out_y * stride - padding + ky * dilation;
                    int in_x = out_x * stride - padding + kx * dilation;
                    
                    // Check bounds
                    if (in_y >= 0 && in_y < height && in_x >= 0 && in_x < width) {
                        int input_idx = batch_idx * (in_channels * height * width) + 
                                       in_ch * (height * width) + 
                                       in_y * width + in_x;
                        int weight_idx = out_ch * (in_channels * kernel_size * kernel_size) + 
                                        in_ch * (kernel_size * kernel_size) + 
                                        ky * kernel_size + kx;
                        conv_result += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        
        // Add bias
        conv_result += bias[out_ch];
        
        // Update minimum
        if (conv_result < min_val) {
            min_val = conv_result;
        }
    }
    
    // Store in shared memory
    int tid = threadIdx.y * blockDim.z + threadIdx.z;
    shared_min[tid] = min_val;
    __syncthreads();
    
    // Reduction in shared memory
    for (int stride = blockDim.y * blockDim.z / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (shared_min[tid + stride] < shared_min[tid]) {
                shared_min[tid] = shared_min[tid + stride];
            }
        }
        __syncthreads();
    }
    
    // Apply double tanh and write result
    if (tid == 0) {
        float result = tanhf(tanhf(shared_min[0]));
        int output_idx = batch_idx * out_height * out_width + 
                        blockIdx.y * blockDim.y * out_width + 
                        blockIdx.z * blockDim.z;
        // Broadcast result to all positions
        for (int i = 0; i < blockDim.y * blockDim.z; i++) {
            if (blockIdx.y * blockDim.y + (i / blockDim.z) < out_height && 
                blockIdx.z * blockDim.z + (i % blockDim.z) < out_width) {
                output[output_idx + (i / blockDim.z) * out_width + (i % blockDim.z)] = result;
            }
        }
    }
}

// Optimized version with better thread organization
__global__ void fused_conv_min_tanh_kernel_v2(
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
    int dilation) {
    
    // Calculate output dimensions
    int out_height = (height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_width = (width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    // Thread indices
    int batch_idx = blockIdx.x;
    int tid_y = threadIdx.y;
    int tid_x = threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + tid_y;
    int out_x = blockIdx.z * blockDim.x + tid_x;
    
    if (batch_idx >= batch_size) return;
    
    // Shared memory for convolution results across channels
    extern __shared__ float shared_results[];
    
    float min_val = INFINITY;
    
    if (out_y < out_height && out_x < out_width) {
        // Perform convolution and find min across all output channels
        for (int out_ch = 0; out_ch < out_channels; out_ch++) {
            float conv_result = 0.0f;
            
            // Perform convolution for this output channel
            for (int in_ch = 0; in_ch < in_channels; in_ch++) {
                for (int ky = 0; ky < kernel_size; ky++) {
                    for (int kx = 0; kx < kernel_size; kx++) {
                        // Calculate input position
                        int in_y = out_y * stride - padding + ky * dilation;
                        int in_x = out_x * stride - padding + kx * dilation;
                        
                        // Check bounds
                        if (in_y >= 0 && in_y < height && in_x >= 0 && in_x < width) {
                            int input_idx = batch_idx * (in_channels * height * width) + 
                                           in_ch * (height * width) + 
                                           in_y * width + in_x;
                            int weight_idx = out_ch * (in_channels * kernel_size * kernel_size) + 
                                            in_ch * (kernel_size * kernel_size) + 
                                            ky * kernel_size + kx;
                            conv_result += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
            
            // Add bias
            conv_result += bias[out_ch];
            
            // Update minimum
            if (conv_result < min_val) {
                min_val = conv_result;
            }
        }
    }
    
    // Apply double tanh to the min value
    float final_result = tanhf(tanhf(min_val));
    
    // Write output
    if (out_y < out_height && out_x < out_width) {
        int output_idx = batch_idx * out_height * out_width + out_y * out_width + out_x;
        output[output_idx] = final_result;
    }
}

void fused_conv_min_tanh_forward(
    const torch::Tensor input,
    const torch::Tensor weight,
    const torch::Tensor bias,
    torch::Tensor output,
    int stride,
    int padding,
    int dilation,
    int groups) {
    
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);
    
    // Calculate output dimensions
    int out_height = (height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_width = (width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    // Launch configuration
    dim3 grid(batch_size, (out_height + 7) / 8, (out_width + 7) / 8);
    dim3 block(8, 8);
    
    // Shared memory size
    size_t shared_mem_size = 64 * sizeof(float); // 8x8 threads
    
    fused_conv_min_tanh_kernel_v2<<<grid, block, shared_mem_size>>>(
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
        dilation
    );
    
    cudaDeviceSynchronize();
}
"""

# C++ binding code
cpp_source = r"""
#include <torch/extension.h>

void fused_conv_min_tanh_forward(
    const torch::Tensor input,
    const torch::Tensor weight,
    const torch::Tensor bias,
    torch::Tensor output,
    int stride,
    int padding,
    int dilation,
    int groups);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_min_tanh", &fused_conv_min_tanh_forward, "Fused Conv-Min-Tanh operation");
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
    out_channels = conv_weight.shape[0]
    kernel_size = conv_weight.shape[2]
    
    # Calculate output dimensions
    out_height = (height + 2 * conv_padding - conv_dilation * (kernel_size - 1) - 1) // conv_stride + 1
    out_width = (width + 2 * conv_padding - conv_dilation * (kernel_size - 1) - 1) // conv_stride + 1
    
    # Create output tensor
    output = torch.empty(batch_size, 1, out_height, out_width, device=x.device, dtype=x.dtype)
    
    # Call the fused CUDA kernel
    fused_ext.fused_conv_min_tanh(
        x, conv_weight, conv_bias, output,
        conv_stride, conv_padding, conv_dilation, conv_groups
    )
    
    return output

# Test parameters
batch_size = 128
in_channels = 16
out_channels = 64
height = width = 256
kernel_size = 3

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]
