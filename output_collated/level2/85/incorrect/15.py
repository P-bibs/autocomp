# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_143406/code_0.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'num_groups', 'scale_shape', 'maxpool_kernel_size', 'clamp_min', 'clamp_max']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_weight', 'conv_bias', 'conv_stride', 'conv_padding', 'conv_dilation', 'conv_groups', 'group_norm_weight', 'group_norm_bias', 'group_norm_num_groups', 'group_norm_eps', 'maxpool_kernel_size', 'maxpool_stride', 'maxpool_padding', 'maxpool_dilation', 'maxpool_ceil_mode', 'maxpool_return_indices', 'scale', 'clamp_min', 'clamp_max']
REQUIRED_FLAT_STATE_NAMES = ['conv_weight', 'conv_bias', 'group_norm_weight', 'group_norm_bias', 'scale']


class ModelNew(nn.Module):
    """
    ModelNew that performs convolution, group normalization, scaling, max pooling, and clamping.
    """

    def __init__(self, in_channels, out_channels, kernel_size, num_groups, scale_shape, maxpool_kernel_size, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.scale = nn.Parameter(torch.ones(scale_shape))
        self.maxpool = nn.MaxPool2d(kernel_size=maxpool_kernel_size)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

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
    # State for group_norm (nn.GroupNorm)
    if 'group_norm_weight' in flat_state:
        state_kwargs['group_norm_weight'] = flat_state['group_norm_weight']
    else:
        state_kwargs['group_norm_weight'] = getattr(model.group_norm, 'weight', None)
    if 'group_norm_bias' in flat_state:
        state_kwargs['group_norm_bias'] = flat_state['group_norm_bias']
    else:
        state_kwargs['group_norm_bias'] = getattr(model.group_norm, 'bias', None)
    state_kwargs['group_norm_num_groups'] = model.group_norm.num_groups
    state_kwargs['group_norm_eps'] = model.group_norm.eps
    # State for maxpool (nn.MaxPool2d)
    state_kwargs['maxpool_kernel_size'] = model.maxpool.kernel_size
    state_kwargs['maxpool_stride'] = model.maxpool.stride
    state_kwargs['maxpool_padding'] = model.maxpool.padding
    state_kwargs['maxpool_dilation'] = model.maxpool.dilation
    state_kwargs['maxpool_ceil_mode'] = model.maxpool.ceil_mode
    state_kwargs['maxpool_return_indices'] = model.maxpool.return_indices
    if 'scale' in flat_state:
        state_kwargs['scale'] = flat_state['scale']
    else:
        state_kwargs['scale'] = getattr(model, 'scale')
    if 'clamp_min' in flat_state:
        state_kwargs['clamp_min'] = flat_state['clamp_min']
    else:
        state_kwargs['clamp_min'] = getattr(model, 'clamp_min')
    if 'clamp_max' in flat_state:
        state_kwargs['clamp_max'] = flat_state['clamp_max']
    else:
        state_kwargs['clamp_max'] = getattr(model, 'clamp_max')
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
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# CUDA kernel that fuses all operations
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

// Function to compute mean and variance for group norm
__device__ void compute_group_stats(
    const float* data,
    int batch_size,
    int channels,
    int height,
    int width,
    int num_groups,
    int group_idx,
    float eps,
    float* mean,
    float* var) {
    
    int channels_per_group = channels / num_groups;
    int start_channel = group_idx * channels_per_group;
    int end_channel = start_channel + channels_per_group;
    
    float sum = 0.0f;
    int count = 0;
    
    for (int b = 0; b < batch_size; b++) {
        for (int c = start_channel; c < end_channel; c++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    int idx = ((b * channels + c) * height + h) * width + w;
                    sum += data[idx];
                    count++;
                }
            }
        }
    }
    
    *mean = sum / count;
    
    float sum_sq_diff = 0.0f;
    for (int b = 0; b < batch_size; b++) {
        for (int c = start_channel; c < end_channel; c++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    int idx = ((b * channels + c) * height + h) * width + w;
                    float diff = data[idx] - *mean;
                    sum_sq_diff += diff * diff;
                }
            }
        }
    }
    
    *var = sum_sq_diff / count + eps;
}

__global__ void fused_op_kernel(
    const float* input,
    float* output,
    const float* conv_weight,
    const float* conv_bias,
    const float* group_norm_weight,
    const float* group_norm_bias,
    float scale,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int kernel_size,
    int conv_stride,
    int conv_padding,
    int num_groups,
    float group_norm_eps,
    int maxpool_kernel_size,
    int maxpool_stride,
    int maxpool_padding,
    float clamp_min,
    float clamp_max,
    int out_height,
    int out_width,
    int pooled_height,
    int pooled_width) {
    
    int tid = threadIdx.x;
    int total_output_elements = batch_size * out_channels * pooled_height * pooled_width;
    
    CUDA_1D_KERNEL_LOOP(idx, total_output_elements) {
        int batch_idx = idx / (out_channels * pooled_height * pooled_width);
        int remaining = idx % (out_channels * pooled_height * pooled_width);
        int channel_idx = remaining / (pooled_height * pooled_width);
        remaining = remaining % (pooled_height * pooled_width);
        int pooled_h_idx = remaining / pooled_width;
        int pooled_w_idx = remaining % pooled_width;
        
        // MaxPool indices
        int h_start = pooled_h_idx * maxpool_stride - maxpool_padding;
        int w_start = pooled_w_idx * maxpool_stride - maxpool_padding;
        
        float max_val = -1e30f;
        
        // Iterate through maxpool window
        for (int ph = 0; ph < maxpool_kernel_size; ++ph) {
            for (int pw = 0; pw < maxpool_kernel_size; ++pw) {
                int h_idx = h_start + ph;
                int w_idx = w_start + pw;
                
                // Check bounds after maxpool padding removal
                if (h_idx >= 0 && h_idx < out_height && w_idx >= 0 && w_idx < out_width) {
                    // Compute conv output at this position
                    float conv_sum = 0.0f;
                    
                    // Convolution computation
                    int conv_h_start = h_idx * conv_stride - conv_padding;
                    int conv_w_start = w_idx * conv_stride - conv_padding;
                    
                    for (int kh = 0; kh < kernel_size; ++kh) {
                        for (int kw = 0; kw < kernel_size; ++kw) {
                            int in_h = conv_h_start + kh;
                            int in_w = conv_w_start + kw;
                            
                            if (in_h >= 0 && in_h < height && in_w >= 0 && in_w < width) {
                                for (int ic = 0; ic < in_channels; ++ic) {
                                    int weight_idx = channel_idx * in_channels * kernel_size * kernel_size + 
                                                    ic * kernel_size * kernel_size + 
                                                    kh * kernel_size + kw;
                                    int input_idx = batch_idx * in_channels * height * width + 
                                                   ic * height * width + 
                                                   in_h * width + in_w;
                                    conv_sum += input[input_idx] * conv_weight[weight_idx];
                                }
                            }
                        }
                    }
                    
                    // Add bias
                    conv_sum += conv_bias[channel_idx];
                    
                    // For simplification in this kernel, we'll use precomputed group statistics
                    // In a real implementation, these would be computed per group
                    float normalized = conv_sum; // Placeholder for actual normalization
                    
                    // Apply group norm scale and bias (per channel)
                    normalized = normalized * group_norm_weight[channel_idx] + group_norm_bias[channel_idx];
                    
                    // Scale
                    normalized = normalized * scale;
                    
                    // Find max in pooling window
                    if (normalized > max_val) {
                        max_val = normalized;
                    }
                }
            }
        }
        
        // Clamp
        if (max_val < clamp_min) max_val = clamp_min;
        if (max_val > clamp_max) max_val = clamp_max;
        
        output[idx] = max_val;
    }
}

// More efficient version that computes conv + norm for entire feature maps first
__global__ void fused_conv_gn_scale_kernel(
    const float* __restrict__ input,
    float* __restrict__ conv_output,
    const float* __restrict__ conv_weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ group_norm_weight,
    const float* __restrict__ group_norm_bias,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int kernel_size,
    int conv_stride,
    int conv_padding,
    int num_groups,
    float group_norm_eps,
    int out_height,
    int out_width) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * out_height * out_width;
    
    if (idx >= total_elements) return;
    
    int batch_idx = idx / (out_channels * out_height * out_width);
    int remaining = idx % (out_channels * out_height * out_width);
    int out_ch = remaining / (out_height * out_width);
    remaining = remaining % (out_height * out_width);
    int out_h = remaining / out_width;
    int out_w = remaining % out_width;
    
    // Convolution
    float conv_sum = 0.0f;
    int in_h_start = out_h * conv_stride - conv_padding;
    int in_w_start = out_w * conv_stride - conv_padding;
    
    for (int kh = 0; kh < kernel_size; ++kh) {
        for (int kw = 0; kw < kernel_size; ++kw) {
            int in_h = in_h_start + kh;
            int in_w = in_w_start + kw;
            
            if (in_h >= 0 && in_h < height && in_w >= 0 && in_w < width) {
                for (int ic = 0; ic < in_channels; ++ic) {
                    int weight_idx = out_ch * in_channels * kernel_size * kernel_size +
                                    ic * kernel_size * kernel_size +
                                    kh * kernel_size + kw;
                    int input_idx = batch_idx * in_channels * height * width +
                                   ic * height * width +
                                   in_h * width + in_w;
                    conv_sum += input[input_idx] * conv_weight[weight_idx];
                }
            }
        }
    }
    
    conv_sum += conv_bias[out_ch];
    conv_output[idx] = conv_sum;
}

__global__ void fused_group_norm_scale_kernel(
    float* __restrict__ conv_output,
    const float* __restrict__ group_norm_weight,
    const float* __restrict__ group_norm_bias,
    float scale,
    int batch_size,
    int channels,
    int height,
    int width,
    int num_groups,
    float eps) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * height * width;
    
    if (idx >= total_elements) return;
    
    int batch_idx = idx / (channels * height * width);
    int remaining = idx % (channels * height * width);
    int ch = remaining / (height * width);
    remaining = remaining % (height * width);
    int h = remaining / width;
    int w = remaining % width;
    
    // Determine which group this channel belongs to
    int channels_per_group = channels / num_groups;
    int group_idx = ch / channels_per_group;
    
    // Compute mean and variance for the group
    float sum = 0.0f;
    float sum_sq = 0.0f;
    int start_ch = group_idx * channels_per_group;
    int end_ch = start_ch + channels_per_group;
    int count = channels_per_group * height * width;
    
    for (int c = start_ch; c < end_ch; c++) {
        for (int gh = 0; gh < height; gh++) {
            for (int gw = 0; gw < width; gw++) {
                int gidx = ((batch_idx * channels + c) * height + gh) * width + gw;
                float val = conv_output[gidx];
                sum += val;
                sum_sq += val * val;
            }
        }
    }
    
    float mean = sum / count;
    float var = (sum_sq / count) - (mean * mean);
    float inv_std = rsqrtf(var + eps);
    
    // Normalize, scale, and shift
    int current_idx = idx;
    float norm_val = (conv_output[current_idx] - mean) * inv_std;
    float scaled_val = norm_val * group_norm_weight[ch] + group_norm_bias[ch];
    conv_output[current_idx] = scaled_val * scale;
}

__global__ void fused_maxpool_clamp_kernel(
    const float* __restrict__ conv_gn_scaled_data,
    float* __restrict__ output,
    int batch_size,
    int channels,
    int height,
    int width,
    int maxpool_kernel_size,
    int maxpool_stride,
    int maxpool_padding,
    float clamp_min,
    float clamp_max,
    int pooled_height,
    int pooled_width) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output_elements = batch_size * channels * pooled_height * pooled_width;
    
    if (idx >= total_output_elements) return;
    
    int batch_idx = idx / (channels * pooled_height * pooled_width);
    int remaining = idx % (channels * pooled_height * pooled_width);
    int ch = remaining / (pooled_height * pooled_width);
    remaining = remaining % (pooled_height * pooled_width);
    int p_h = remaining / pooled_width;
    int p_w = remaining % pooled_width;
    
    // MaxPool indices
    int h_start = p_h * maxpool_stride - maxpool_padding;
    int w_start = p_w * maxpool_stride - maxpool_padding;
    
    float max_val = -1e30f;
    
    // Iterate through maxpool window
    for (int ph = 0; ph < maxpool_kernel_size; ++ph) {
        for (int pw = 0; pw < maxpool_kernel_size; ++pw) {
            int h_idx = h_start + ph;
            int w_idx = w_start + pw;
            
            // Check bounds
            if (h_idx >= 0 && h_idx < height && w_idx >= 0 && w_idx < width) {
                int input_idx = ((batch_idx * channels + ch) * height + h_idx) * width + w_idx;
                float val = conv_gn_scaled_data[input_idx];
                
                if (val > max_val) {
                    max_val = val;
                }
            }
        }
    }
    
    // Clamp
    if (max_val < clamp_min) max_val = clamp_min;
    if (max_val > clamp_max) max_val = clamp_max;
    
    output[idx] = max_val;
}

void fused_op_forward(
    const at::Tensor& input,
    at::Tensor& output,
    at::Tensor& temp_buffer,
    const at::Tensor& conv_weight,
    const at::Tensor& conv_bias,
    const at::Tensor& group_norm_weight,
    const at::Tensor& group_norm_bias,
    float scale,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int kernel_size,
    int conv_stride,
    int conv_padding,
    int num_groups,
    float group_norm_eps,
    int maxpool_kernel_size,
    int maxpool_stride,
    int maxpool_padding,
    float clamp_min,
    float clamp_max,
    int out_height,
    int out_width,
    int pooled_height,
    int pooled_width) {
    
    // Process in stages for better memory efficiency
    
    // Stage 1: Convolution
    int conv_elements = batch_size * out_channels * out_height * out_width;
    int threads_per_block = 256;
    int blocks = (conv_elements + threads_per_block - 1) / threads_per_block;
    blocks = min(blocks, 65535);
    
    fused_conv_gn_scale_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        temp_buffer.data_ptr<float>(),
        conv_weight.data_ptr<float>(),
        conv_bias.data_ptr<float>(),
        group_norm_weight.data_ptr<float>(),
        group_norm_bias.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        kernel_size,
        conv_stride,
        conv_padding,
        num_groups,
        group_norm_eps,
        out_height,
        out_width
    );
    
    // Stage 2: Group Norm + Scale
    fused_group_norm_scale_kernel<<<blocks, threads_per_block>>>(
        temp_buffer.data_ptr<float>(),
        group_norm_weight.data_ptr<float>(),
        group_norm_bias.data_ptr<float>(),
        scale,
        batch_size,
        out_channels,
        out_height,
        out_width,
        num_groups,
        group_norm_eps
    );
    
    // Stage 3: MaxPool + Clamp
    int pool_elements = batch_size * out_channels * pooled_height * pooled_width;
    blocks = (pool_elements + threads_per_block - 1) / threads_per_block;
    blocks = min(blocks, 65535);
    
    fused_maxpool_clamp_kernel<<<blocks, threads_per_block>>>(
        temp_buffer.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        out_channels,
        out_height,
        out_width,
        maxpool_kernel_size,
        maxpool_stride,
        maxpool_padding,
        clamp_min,
        clamp_max,
        pooled_height,
        pooled_width
    );
    
    cudaDeviceSynchronize();
}
"""

# C++ binding code
cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(
    const at::Tensor& input,
    at::Tensor& output,
    at::Tensor& temp_buffer,
    const at::Tensor& conv_weight,
    const at::Tensor& conv_bias,
    const at::Tensor& group_norm_weight,
    const at::Tensor& group_norm_bias,
    float scale,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int kernel_size,
    int conv_stride,
    int conv_padding,
    int num_groups,
    float group_norm_eps,
    int maxpool_kernel_size,
    int maxpool_stride,
    int maxpool_padding,
    float clamp_min,
    float clamp_max,
    int out_height,
    int out_width,
    int pooled_height,
    int pooled_width);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused Conv + GroupNorm + Scale + MaxPool + Clamp operation");
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

# Optimized functional model using fused kernel
def functional_model(
    x,
    *,
    conv_weight,
    conv_bias,
    conv_stride,
    conv_padding,
    conv_dilation,
    conv_groups,
    group_norm_weight,
    group_norm_bias,
    group_norm_num_groups,
    group_norm_eps,
    maxpool_kernel_size,
    maxpool_stride,
    maxpool_padding,
    maxpool_dilation,
    maxpool_ceil_mode,
    maxpool_return_indices,
    scale,
    clamp_min,
    clamp_max,
):
    # Validate inputs
    assert conv_dilation == 1, "Only dilation=1 supported"
    assert conv_groups == 1, "Only groups=1 supported"
    assert maxpool_dilation == 1, "Only maxpool_dilation=1 supported"
    assert not maxpool_return_indices, "maxpool_return_indices not supported"
    
    batch_size, in_channels, height, width = x.shape
    out_channels = conv_weight.shape[0]
    kernel_size = conv_weight.shape[2]
    
    # Calculate output dimensions for convolution
    out_height = (height + 2 * conv_padding - kernel_size) // conv_stride + 1
    out_width = (width + 2 * conv_padding - kernel_size) // conv_stride + 1
    
    # Calculate pooled dimensions
    if maxpool_ceil_mode:
        pooled_height = (out_height + 2 * maxpool_padding - maxpool_kernel_size + maxpool_stride - 1) // maxpool_stride + 1
        pooled_width = (out_width + 2 * maxpool_padding - maxpool_kernel_size + maxpool_stride - 1) // maxpool_stride + 1
    else:
        pooled_height = (out_height + 2 * maxpool_padding - maxpool_kernel_size) // maxpool_stride + 1
        pooled_width = (out_width + 2 * maxpool_padding - maxpool_kernel_size) // maxpool_stride + 1
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, pooled_height, pooled_width, device=x.device, dtype=x.dtype)
    
    # Create temporary buffer for intermediate results
    temp_buffer = torch.empty(batch_size, out_channels, out_height, out_width, device=x.device, dtype=x.dtype)
    
    # Call fused kernel
    fused_ext.fused_op(
        x, output, temp_buffer,
        conv_weight, conv_bias,
        group_norm_weight, group_norm_bias,
        scale,
        batch_size, in_channels, out_channels,
        height, width, kernel_size,
        conv_stride, conv_padding,
        group_norm_num_groups, group_norm_eps,
        maxpool_kernel_size, maxpool_stride, maxpool_padding,
        clamp_min, clamp_max,
        out_height, out_width,
        pooled_height, pooled_width
    )
    
    return output

# Test parameters
batch_size = 128
in_channels = 8
out_channels = 64
height, width = 128, 128 
kernel_size = 3
num_groups = 16
scale_shape = (out_channels, 1, 1)
maxpool_kernel_size = 4
clamp_min = 0.0
clamp_max = 1.0

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, num_groups, scale_shape, maxpool_kernel_size, clamp_min, clamp_max]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]
