# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_144547/code_0.py
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define CUDA_1D_KERNEL_LOOP(i, n) \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)

// Device function to compute convolution at a specific output position
__device__ float compute_conv2d(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    int batch,
    int out_ch,
    int out_h,
    int out_w,
    int in_channels,
    int in_height,
    int in_width,
    int kernel_size,
    int stride,
    int padding,
    int weight_offset
) {
    float sum = 0.0f;
    
    for (int kh = 0; kh < kernel_size; ++kh) {
        for (int kw = 0; kw < kernel_size; ++kw) {
            int in_h = out_h * stride - padding + kh;
            int in_w = out_w * stride - padding + kw;
            
            if (in_h >= 0 && in_h < in_height && in_w >= 0 && in_w < in_width) {
                for (int in_ch = 0; in_ch < in_channels; ++in_ch) {
                    int in_idx = batch * (in_channels * in_height * in_width) +
                                in_ch * (in_height * in_width) + 
                                in_h * in_width + in_w;
                    int weight_idx = weight_offset + 
                                   in_ch * (kernel_size * kernel_size) + 
                                   kh * kernel_size + kw;
                    sum += input[in_idx] * weight[weight_idx];
                }
            }
        }
    }
    return sum + bias[out_ch];
}

// Device function to compute group normalization
__device__ float compute_group_norm(
    float value,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    int ch,
    int num_groups,
    int num_channels
) {
    int group = ch * num_groups / num_channels;
    return value * weight[ch] + bias[ch];
}

// Device function to compute max pooling with window size
__device__ float compute_max_pool2d(
    const float* __restrict__ input,
    const float* __restrict__ conv_weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ gn_weight,
    const float* __restrict__ gn_bias,
    const float* __restrict__ x,
    int batch,
    int out_ch,
    int pool_out_h,
    int pool_out_w,
    int conv_out_h,
    int conv_out_w,
    int in_channels,
    int in_height,
    int in_width,
    int kernel_size,
    int conv_stride,
    int conv_padding,
    int pool_kernel_size,
    int pool_stride,
    int pool_padding,
    int num_groups,
    float scale
) {
    float max_val = -1e30f;
    
    // Calculate the starting position of the pooling window in conv space
    int conv_start_h = pool_out_h * pool_stride - pool_padding;
    int conv_start_w = pool_out_w * pool_stride - pool_padding;
    
    for (int ph = 0; ph < pool_kernel_size; ++ph) {
        for (int pw = 0; pw < pool_kernel_size; ++pw) {
            int conv_h = conv_start_h + ph;
            int conv_w = conv_start_w + pw;
            
            if (conv_h >= 0 && conv_h < conv_out_h && 
                conv_w >= 0 && conv_w < conv_out_w) {
                
                // Compute conv at this position
                float conv_val = compute_conv2d(
                    x, conv_weight, conv_bias,
                    batch, out_ch, conv_h, conv_w,
                    in_channels, in_height, in_width,
                    kernel_size, conv_stride, conv_padding,
                    out_ch * in_channels * kernel_size * kernel_size
                );
                
                // Apply group normalization
                float gn_val = compute_group_norm(conv_val, gn_weight, gn_bias, out_ch, num_groups, gridDim.x);
                
                // Apply scale
                float scaled_val = gn_val * scale;
                
                max_val = fmaxf(max_val, scaled_val);
            }
        }
    }
    
    return max_val;
}

__global__ void fused_conv_gn_scale_pool_clamp_kernel(
    const float* __restrict__ x,
    const float* __restrict__ conv_weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ gn_weight,
    const float* __restrict__ gn_bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_height,
    int in_width,
    int kernel_size,
    int conv_stride,
    int conv_padding,
    int num_groups,
    float scale,
    int pool_kernel_size,
    int pool_stride,
    int pool_padding,
    float clamp_min,
    float clamp_max,
    int conv_out_height,
    int conv_out_width,
    int pool_out_height,
    int pool_out_width
) {
    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * pool_out_height * pool_out_width;
    
    if (idx >= total_elements) return;
    
    // Decompose linear index into 4D coordinates
    int pool_w = idx % pool_out_width;
    int pool_h = (idx / pool_out_width) % pool_out_height;
    int ch = (idx / (pool_out_width * pool_out_height)) % out_channels;
    int batch = idx / (pool_out_width * pool_out_height * out_channels);
    
    // Compute max pooling with all fused operations
    float result = compute_max_pool2d(
        output, conv_weight, conv_bias, gn_weight, gn_bias, x,
        batch, ch, pool_h, pool_w,
        conv_out_height, conv_out_width,
        in_channels, in_height, in_width,
        kernel_size, conv_stride, conv_padding,
        pool_kernel_size, pool_stride, pool_padding,
        num_groups, scale
    );
    
    // Apply clamp
    result = fmaxf(clamp_min, fminf(clamp_max, result));
    
    // Store final result
    output[idx] = result;
}

void fused_model_forward(
    torch::Tensor x,
    torch::Tensor conv_weight,
    torch::Tensor conv_bias,
    torch::Tensor gn_weight,
    torch::Tensor gn_bias,
    torch::Tensor out,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_height,
    int in_width,
    int kernel_size,
    int conv_stride,
    int conv_padding,
    int num_groups,
    float scale,
    int pool_kernel_size,
    int pool_stride,
    int pool_padding,
    float clamp_min,
    float clamp_max,
    int conv_out_height,
    int conv_out_width,
    int pool_out_height,
    int pool_out_width
) {
    int total_threads = batch_size * out_channels * pool_out_height * pool_out_width;
    int threads_per_block = 256;
    int blocks = (total_threads + threads_per_block - 1) / threads_per_block;
    
    fused_conv_gn_scale_pool_clamp_kernel<<<blocks, threads_per_block>>>(
        x.data_ptr<float>(),
        conv_weight.data_ptr<float>(),
        conv_bias.data_ptr<float>(),
        gn_weight.data_ptr<float>(),
        gn_bias.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        in_height,
        in_width,
        kernel_size,
        conv_stride,
        conv_padding,
        num_groups,
        scale,
        pool_kernel_size,
        pool_stride,
        pool_padding,
        clamp_min,
        clamp_max,
        conv_out_height,
        conv_out_width,
        pool_out_height,
        pool_out_width
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_model_forward(
    torch::Tensor x,
    torch::Tensor conv_weight,
    torch::Tensor conv_bias,
    torch::Tensor gn_weight,
    torch::Tensor gn_bias,
    torch::Tensor out,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_height,
    int in_width,
    int kernel_size,
    int conv_stride,
    int conv_padding,
    int num_groups,
    float scale,
    int pool_kernel_size,
    int pool_stride,
    int pool_padding,
    float clamp_min,
    float clamp_max,
    int conv_out_height,
    int conv_out_width,
    int pool_out_height,
    int pool_out_width);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_model_forward", &fused_model_forward, "Fused model forward pass");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_model_ext',
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
    batch_size, in_channels, height, width = x.shape
    out_channels = conv_weight.shape[0]
    
    # Calculate convolution output dimensions
    conv_out_height = (height + 2 * conv_padding[0] - conv_dilation[0] * (conv_weight.shape[2] - 1) - 1) // conv_stride[0] + 1
    conv_out_width = (width + 2 * conv_padding[1] - conv_dilation[1] * (conv_weight.shape[3] - 1) - 1) // conv_stride[1] + 1
    
    # Calculate pooling output dimensions
    pool_out_height = (conv_out_height + 2 * maxpool_padding[0] - maxpool_dilation[0] * (maxpool_kernel_size - 1) - 1) // maxpool_stride[0] + 1
    pool_out_width = (conv_out_width + 2 * maxpool_padding[1] - maxpool_dilation[1] * (maxpool_kernel_size - 1) - 1) // maxpool_stride[1] + 1
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, pool_out_height, pool_out_width, device=x.device, dtype=x.dtype)
    
    # Call fused kernel
    fused_ext.fused_model_forward(
        x.contiguous(),
        conv_weight.contiguous(),
        conv_bias.contiguous(),
        group_norm_weight.contiguous(),
        group_norm_bias.contiguous(),
        output,
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        conv_weight.shape[2],  # kernel_size
        conv_stride[0],
        conv_padding[0],
        group_norm_num_groups,
        float(scale),
        maxpool_kernel_size,
        maxpool_stride[0],
        maxpool_padding[0],
        float(clamp_min),
        float(clamp_max),
        conv_out_height,
        conv_out_width,
        pool_out_height,
        pool_out_width
    )
    
    return output

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
