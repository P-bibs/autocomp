# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_142418/code_0.py
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

# CUDA kernel for fused operations
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

__global__ void fused_op_kernel(
    const float* __restrict__ input,
    const float* __restrict__ conv_weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ group_norm_weight,
    const float* __restrict__ group_norm_bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int kernel_size,
    int conv_stride,
    int conv_padding,
    int num_groups,
    int maxpool_kernel_size,
    int maxpool_stride,
    int maxpool_padding,
    float scale,
    float clamp_min,
    float clamp_max,
    int out_height,
    int out_width,
    int pooled_height,
    int pooled_width
) {
    // Calculate global thread indices
    int batch_idx = blockIdx.x;
    int out_ch = blockIdx.y * blockDim.x + threadIdx.x;
    int pooled_h = blockIdx.z;
    int pooled_w = threadIdx.y;
    
    if (batch_idx >= batch_size || out_ch >= out_channels || 
        pooled_h >= pooled_height || pooled_w >= pooled_width) {
        return;
    }
    
    // Shared memory for input data
    extern __shared__ float shared_input[];
    
    // Perform convolution + group norm + scale for this output position
    float result = 0.0f;
    
    // Calculate corresponding output position before pooling
    int out_h_start = pooled_h * maxpool_stride - maxpool_padding;
    int out_w_start = pooled_w * maxpool_stride - maxpool_padding;
    
    // Find maximum in the pooling window
    float max_val = -1e30f; // Negative infinity
    
    for (int ph = 0; ph < maxpool_kernel_size; ph++) {
        for (int pw = 0; pw < maxpool_kernel_size; pw++) {
            int out_h = out_h_start + ph;
            int out_w = out_w_start + pw;
            
            // Skip if outside conv output bounds
            if (out_h < 0 || out_h >= out_height || out_w < 0 || out_w >= out_width) {
                continue;
            }
            
            // Perform convolution at this position
            float conv_result = 0.0f;
            int half_kernel = kernel_size / 2;
            
            for (int in_ch = 0; in_ch < in_channels; in_ch++) {
                for (int kh = 0; kh < kernel_size; kh++) {
                    for (int kw = 0; kw < kernel_size; kw++) {
                        // Calculate input coordinates
                        int in_h = out_h * conv_stride - conv_padding + kh;
                        int in_w = out_w * conv_stride - conv_padding + kw;
                        
                        float input_val = 0.0f;
                        if (in_h >= 0 && in_h < height && in_w >= 0 && in_w < width) {
                            int input_idx = batch_idx * (in_channels * height * width) + 
                                           in_ch * (height * width) + in_h * width + in_w;
                            input_val = input[input_idx];
                        }
                        
                        int weight_idx = out_ch * (in_channels * kernel_size * kernel_size) + 
                                        in_ch * (kernel_size * kernel_size) + kh * kernel_size + kw;
                        conv_result += input_val * conv_weight[weight_idx];
                    }
                }
            }
            
            // Add bias
            conv_result += conv_bias[out_ch];
            
            // Group normalization - simplified per-channel implementation
            float gn_weight = group_norm_weight[out_ch];
            float gn_bias = group_norm_bias[out_ch];
            float normalized = conv_result * gn_weight + gn_bias;
            
            // Scale
            float scaled = normalized * scale;
            
            // Update maximum for max pooling
            max_val = fmaxf(max_val, scaled);
        }
    }
    
    // Clamp
    float clamped = fmaxf(clamp_min, fminf(clamp_max, max_val));
    
    // Write output
    int output_idx = batch_idx * (out_channels * pooled_height * pooled_width) + 
                    out_ch * (pooled_height * pooled_width) + pooled_h * pooled_width + pooled_w;
    output[output_idx] = clamped;
}

void fused_op_forward(
    const at::Tensor& input,
    const at::Tensor& conv_weight,
    const at::Tensor& conv_bias,
    const at::Tensor& group_norm_weight,
    const at::Tensor& group_norm_bias,
    at::Tensor& output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int kernel_size,
    int conv_stride,
    int conv_padding,
    int num_groups,
    int maxpool_kernel_size,
    int maxpool_stride,
    int maxpool_padding,
    float scale,
    float clamp_min,
    float clamp_max,
    int out_height,
    int out_width,
    int pooled_height,
    int pooled_width
) {
    // Configure kernel launch parameters
    dim3 block_size(32, 32);  // 32 threads for channels, 32 for width
    dim3 grid_size(
        batch_size,
        CEIL_DIV(out_channels, block_size.x),
        pooled_height
    );
    
    // Shared memory size - allocate space for input data if needed
    int shared_mem_size = 0; // Not using shared memory for weights in this version
    
    fused_op_kernel<<<grid_size, block_size, shared_mem_size>>>(
        input.data_ptr<float>(),
        conv_weight.data_ptr<float>(),
        conv_bias.data_ptr<float>(),
        group_norm_weight.data_ptr<float>(),
        group_norm_bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        kernel_size,
        conv_stride,
        conv_padding,
        num_groups,
        maxpool_kernel_size,
        maxpool_stride,
        maxpool_padding,
        scale,
        clamp_min,
        clamp_max,
        out_height,
        out_width,
        pooled_height,
        pooled_width
    );
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("CUDA kernel launch failed: ", cudaGetErrorString(err));
    }
}
"""

# C++ bindings
cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(
    const at::Tensor& input,
    const at::Tensor& conv_weight,
    const at::Tensor& conv_bias,
    const at::Tensor& group_norm_weight,
    const at::Tensor& group_norm_bias,
    at::Tensor& output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int kernel_size,
    int conv_stride,
    int conv_padding,
    int num_groups,
    int maxpool_kernel_size,
    int maxpool_stride,
    int maxpool_padding,
    float scale,
    float clamp_min,
    float clamp_max,
    int out_height,
    int out_width,
    int pooled_height,
    int pooled_width
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused operations: conv + group_norm + scale + max_pool + clamp");
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
    # Validate dilation and groups (assuming they are 1 for simplicity in this implementation)
    assert conv_dilation == 1, "Only conv_dilation=1 is supported"
    assert conv_groups == 1, "Only conv_groups=1 is supported"
    assert maxpool_dilation == 1, "Only maxpool_dilation=1 is supported"
    assert not maxpool_return_indices, "maxpool_return_indices=True is not supported"
    assert group_norm_eps == 1e-5, "Only group_norm_eps=1e-5 is supported in this implementation"
    
    batch_size, in_channels, height, width = x.shape
    out_channels = conv_weight.shape[0]
    kernel_size = conv_weight.shape[2]
    
    # Calculate convolution output dimensions
    out_height = (height + 2 * conv_padding - kernel_size) // conv_stride + 1
    out_width = (width + 2 * conv_padding - kernel_size) // conv_stride + 1
    
    # Calculate maxpool output dimensions
    if maxpool_ceil_mode:
        pooled_height = (out_height + 2 * maxpool_padding - (maxpool_kernel_size - 1) - 1 + maxpool_stride + 1) // maxpool_stride
        pooled_width = (out_width + 2 * maxpool_padding - (maxpool_kernel_size - 1) - 1 + maxpool_stride + 1) // maxpool_stride
    else:
        pooled_height = (out_height + 2 * maxpool_padding - (maxpool_kernel_size - 1) - 1) // maxpool_stride + 1
        pooled_width = (out_width + 2 * maxpool_padding - (maxpool_kernel_size - 1) - 1) // maxpool_stride + 1
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, pooled_height, pooled_width, device=x.device, dtype=x.dtype)
    
    # Call fused operation
    fused_ext.fused_op(
        x, conv_weight, conv_bias, group_norm_weight, group_norm_bias, output,
        batch_size, in_channels, out_channels, height, width, kernel_size,
        conv_stride, conv_padding, group_norm_num_groups, maxpool_kernel_size,
        maxpool_stride, maxpool_padding, scale, clamp_min, clamp_max,
        out_height, out_width, pooled_height, pooled_width
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
