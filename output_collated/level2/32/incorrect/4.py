# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_011257/code_2.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'scale_factor']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_weight', 'conv_bias', 'conv_stride', 'conv_padding', 'conv_dilation', 'conv_groups', 'scale_factor']
REQUIRED_FLAT_STATE_NAMES = ['conv_weight', 'conv_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a convolution, scales the output, and then applies a minimum operation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor

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
    if 'scale_factor' in flat_state:
        state_kwargs['scale_factor'] = flat_state['scale_factor']
    else:
        state_kwargs['scale_factor'] = getattr(model, 'scale_factor')
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

# CUDA kernel for fused convolution, scaling and min reduction
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

__global__ void fused_conv_scale_min_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float scale_factor,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int height,
    const int width,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation,
    const int groups
) {
    // Calculate output dimensions
    const int out_height = (height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int out_width = (width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    // Get thread indices
    const int batch_idx = blockIdx.x;
    const int out_ch_idx = blockIdx.y;
    const int tid = threadIdx.x;
    
    if (batch_idx >= batch_size || out_ch_idx >= out_channels) return;
    
    // Shared memory for reduction
    extern __shared__ float sdata[];
    
    // Each thread processes multiple output positions
    const int total_outputs = out_height * out_width;
    const int threads_per_block = blockDim.x;
    
    float thread_min = FLT_MAX;
    
    // Process multiple output positions per thread
    for (int idx = tid; idx < total_outputs; idx += threads_per_block) {
        const int out_y = idx / out_width;
        const int out_x = idx % out_width;
        
        float conv_result = 0.0f;
        
        // Perform convolution for this output position
        for (int g = 0; g < groups; g++) {
            if (out_ch_idx >= (g + 1) * (out_channels / groups)) continue;
            if (out_ch_idx < g * (out_channels / groups)) continue;
            
            const int in_ch_start = g * (in_channels / groups);
            const int in_ch_end = (g + 1) * (in_channels / groups);
            const int weight_out_ch_offset = out_ch_idx * (in_channels / groups) * kernel_size * kernel_size;
            
            for (int in_ch = in_ch_start; in_ch < in_ch_end; in_ch++) {
                const int weight_offset = weight_out_ch_offset + (in_ch - in_ch_start) * kernel_size * kernel_size;
                const int input_ch_offset = batch_idx * in_channels * height * width + in_ch * height * width;
                
                for (int ky = 0; ky < kernel_size; ky++) {
                    for (int kx = 0; kx < kernel_size; kx++) {
                        const int in_y = out_y * stride - padding + ky * dilation;
                        const int in_x = out_x * stride - padding + kx * dilation;
                        
                        if (in_y >= 0 && in_y < height && in_x >= 0 && in_x < width) {
                            const int input_idx = input_ch_offset + in_y * width + in_x;
                            const int weight_idx = weight_offset + ky * kernel_size + kx;
                            conv_result += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
        
        // Add bias
        conv_result += bias[out_ch_idx];
        
        // Scale and update minimum
        const float scaled_value = conv_result * scale_factor;
        thread_min = fminf(thread_min, scaled_value);
    }
    
    // Store thread result in shared memory
    sdata[tid] = thread_min;
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fminf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    
    // Write result
    if (tid == 0) {
        output[batch_idx * out_height * out_width + blockIdx.y * out_height * out_width] = sdata[0];
    }
}

// Kernel for the final min reduction across channels
__global__ void final_min_reduction_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int batch_size,
    const int out_channels,
    const int out_height,
    const int out_width
) {
    const int batch_idx = blockIdx.x;
    const int spatial_idx = blockIdx.y * blockDim.x + threadIdx.x;
    
    const int total_spatial = out_height * out_width;
    if (batch_idx >= batch_size || spatial_idx >= total_spatial) return;
    
    const int out_y = spatial_idx / out_width;
    const int out_x = spatial_idx % out_width;
    
    float min_val = FLT_MAX;
    for (int ch = 0; ch < out_channels; ch++) {
        const int idx = batch_idx * out_channels * out_height * out_width + 
                        ch * out_height * out_width + 
                        out_y * out_width + out_x;
        min_val = fminf(min_val, input[idx]);
    }
    
    const int out_idx = batch_idx * out_height * out_width + out_y * out_width + out_x;
    output[out_idx] = min_val;
}

void fused_conv_scale_min_forward(
    const torch::Tensor input,
    const torch::Tensor weight,
    const torch::Tensor bias,
    const float scale_factor,
    torch::Tensor output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int height,
    const int width,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation,
    const int groups
) {
    const int out_height = (height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int out_width = (width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    // First kernel: convolution + scaling
    const dim3 blocks_conv(batch_size, out_channels);
    const dim3 threads_conv(256);
    const int shared_mem_size = 256 * sizeof(float);
    
    fused_conv_scale_min_kernel<<<blocks_conv, threads_conv, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        scale_factor,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        kernel_size,
        stride,
        padding,
        dilation,
        groups
    );
    
    // Second kernel: min reduction across channels
    const dim3 blocks_reduction(batch_size, (out_height * out_width + 255) / 256);
    const dim3 threads_reduction(256);
    
    // Temporary tensor for intermediate results
    auto temp = torch::empty({batch_size, out_channels, out_height, out_width}, 
                             torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    
    final_min_reduction_kernel<<<blocks_reduction, threads_reduction>>>(
        temp.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        out_channels,
        out_height,
        out_width
    );
}
"""

# C++ interface
cpp_source = r"""
#include <torch/extension.h>

void fused_conv_scale_min_forward(
    const torch::Tensor input,
    const torch::Tensor weight,
    const torch::Tensor bias,
    const float scale_factor,
    torch::Tensor output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int height,
    const int width,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation,
    const int groups
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_scale_min", &fused_conv_scale_min_forward, "Fused Convolution, Scale and Min Reduction");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_conv_scale_min',
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
    scale_factor,
):
    batch_size, in_channels, height, width = x.shape
    out_channels = conv_weight.shape[0]
    kernel_size = conv_weight.shape[2]
    
    # Calculate output dimensions
    out_height = (height + 2 * conv_padding - conv_dilation * (kernel_size - 1) - 1) // conv_stride + 1
    out_width = (width + 2 * conv_padding - conv_dilation * (kernel_size - 1) - 1) // conv_stride + 1
    
    # Create output tensor with correct shape [batch_size, 1, out_height, out_width]
    output = torch.empty(batch_size, 1, out_height, out_width, device=x.device, dtype=x.dtype)
    
    # Call fused kernel
    fused_ext.fused_conv_scale_min(
        x, conv_weight, conv_bias, scale_factor, output,
        batch_size, in_channels, out_channels, height, width,
        kernel_size, conv_stride, conv_padding, conv_dilation, conv_groups
    )
    
    return output

batch_size = 64
in_channels = 64
out_channels = 128
height = width = 256
kernel_size = 3
scale_factor = 2.0

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scale_factor]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]
