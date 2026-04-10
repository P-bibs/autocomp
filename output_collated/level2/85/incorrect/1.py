# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_141453/code_2.py
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

# --- CUDA Kernel Code ---
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_scale_pool_clamp_kernel(
    const float* input,
    float* output,
    const float scale,
    const int batch_size,
    const int channels,
    const int input_height,
    const int input_width,
    const int pool_kernel_size,
    const int pool_stride,
    const int pool_padding,
    const float clamp_min,
    const float clamp_max
) {
    // Calculate output dimensions
    const int output_height = (input_height + 2 * pool_padding - pool_kernel_size) / pool_stride + 1;
    const int output_width = (input_width + 2 * pool_padding - pool_kernel_size) / pool_stride + 1;

    // Get indices
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * channels * output_height * output_width;
    
    if (tid >= total_elements) return;

    // Decompose linear index into 4D coordinates
    const int ow = tid % output_width;
    const int oh = (tid / output_width) % output_height;
    const int c = (tid / (output_width * output_height)) % channels;
    const int b = tid / (output_width * output_height * channels);

    // Calculate input window start coordinates
    const int ih_start = oh * pool_stride - pool_padding;
    const int iw_start = ow * pool_stride - pool_padding;

    // Initialize max value
    float max_val = -INFINITY;

    // Iterate through pooling window
    for (int kh = 0; kh < pool_kernel_size; ++kh) {
        const int ih = ih_start + kh;
        if (ih < 0 || ih >= input_height) continue;

        for (int kw = 0; kw < pool_kernel_size; ++kw) {
            const int iw = iw_start + kw;
            if (iw < 0 || iw >= input_width) continue;

            // Calculate input index
            const int input_idx = ((b * channels + c) * input_height + ih) * input_width + iw;
            const float val = input[input_idx] * scale; // Apply scaling here
            max_val = fmaxf(max_val, val);
        }
    }

    // Clamp the result
    max_val = fminf(fmaxf(max_val, clamp_min), clamp_max);

    // Write to output
    output[tid] = max_val;
}

// Conv2D kernel
__global__ void conv2d_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int input_height,
    const int input_width,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation
) {
    const int output_height = (input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int output_width = (input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * out_channels * output_height * output_width;

    if (tid >= total_elements) return;

    const int ow = tid % output_width;
    const int oh = (tid / output_width) % output_height;
    const int oc = (tid / (output_width * output_height)) % out_channels;
    const int b = tid / (output_width * output_height * out_channels);

    float sum = 0.0f;

    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                const int ih = oh * stride - padding + kh * dilation;
                const int iw = ow * stride - padding + kw * dilation;

                if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                    const int input_idx = ((b * in_channels + ic) * input_height + ih) * input_width + iw;
                    const int weight_idx = ((oc * in_channels + ic) * kernel_size + kh) * kernel_size + kw;
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    const int output_idx = tid;
    output[output_idx] = sum + bias[oc];
}

// GroupNorm kernel
__global__ void group_norm_kernel(
    const float* input,
    float* output,
    const float* weight,
    const float* bias,
    const int batch_size,
    const int num_channels,
    const int height,
    const int width,
    const int num_groups,
    const float eps
) {
    const int group_size = num_channels / num_groups;
    const int group_idx = blockIdx.x;
    const int tid = threadIdx.x;
    
    if (group_idx >= num_groups) return;

    const int elements_per_group = group_size * height * width;
    const int start_channel = group_idx * group_size;
    
    // Shared memory for mean and variance calculation
    extern __shared__ float shared_mem[];
    float* shared_data = shared_mem;
    float* mean_shared = shared_mem + elements_per_group;
    float* var_shared = mean_shared + 1;
    
    // Initialize shared memory
    if (tid == 0) {
        mean_shared[0] = 0.0f;
        var_shared[0] = 0.0f;
    }
    __syncthreads();
    
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    int count = 0;
    
    // Calculate sum and sum of squares
    for (int i = tid; i < elements_per_group; i += blockDim.x) {
        const int c = start_channel + (i / (height * width));
        const int h = (i / width) % height;
        const int w = i % width;
        
        if (c < num_channels) {
            const int idx = ((blockIdx.y * num_channels + c) * height + h) * width + w;
            const float val = input[idx];
            local_sum += val;
            local_sum_sq += val * val;
            count++;
        }
    }
    
    // Reduce within block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        __syncthreads();
        if (tid < stride) {
            local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, stride);
            local_sum_sq += __shfl_down_sync(0xFFFFFFFF, local_sum_sq, stride);
        }
    }
    
    if (tid == 0) {
        atomicAdd(mean_shared, local_sum);
        atomicAdd(var_shared, local_sum_sq);
    }
    __syncthreads();
    
    const float mean = mean_shared[0] / (group_size * height * width);
    const float var = var_shared[0] / (group_size * height * width) - mean * mean;
    const float inv_std = rsqrtf(var + eps);
    
    // Normalize and apply affine transformation
    for (int i = tid; i < elements_per_group; i += blockDim.x) {
        const int c = start_channel + (i / (height * width));
        const int h = (i / width) % height;
        const int w = i % width;
        
        if (c < num_channels) {
            const int idx = ((blockIdx.y * num_channels + c) * height + h) * width + w;
            const float normalized = (input[idx] - mean) * inv_std;
            const int weight_idx = c;
            output[idx] = normalized * weight[weight_idx] + bias[weight_idx];
        }
    }
}

void fused_scale_pool_clamp_op(
    const torch::Tensor input,
    torch::Tensor output,
    const float scale,
    const int pool_kernel_size,
    const int pool_stride,
    const int pool_padding,
    const float clamp_min,
    const float clamp_max
) {
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int input_height = input.size(2);
    const int input_width = input.size(3);
    
    const int output_height = (input_height + 2 * pool_padding - pool_kernel_size) / pool_stride + 1;
    const int output_width = (input_width + 2 * pool_padding - pool_kernel_size) / pool_stride + 1;
    
    const int total_elements = batch_size * channels * output_height * output_width;
    const int threads_per_block = 256;
    const int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    fused_scale_pool_clamp_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        scale,
        batch_size,
        channels,
        input_height,
        input_width,
        pool_kernel_size,
        pool_stride,
        pool_padding,
        clamp_min,
        clamp_max
    );
}

void conv2d_op(
    const torch::Tensor input,
    const torch::Tensor weight,
    const torch::Tensor bias,
    torch::Tensor output,
    const int stride,
    const int padding,
    const int dilation
) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int input_height = input.size(2);
    const int input_width = input.size(3);
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);
    
    const int output_height = (input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int output_width = (input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    const int total_elements = batch_size * out_channels * output_height * output_width;
    const int threads_per_block = 256;
    const int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    conv2d_kernel<<<blocks, threads_per_block>>>(
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
        dilation
    );
}

void group_norm_op(
    const torch::Tensor input,
    torch::Tensor output,
    const torch::Tensor weight,
    const torch::Tensor bias,
    const int num_groups,
    const float eps
) {
    const int batch_size = input.size(0);
    const int num_channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);
    
    const int group_size = num_channels / num_groups;
    const int elements_per_group = group_size * height * width;
    
    const int threads_per_block = 256;
    const int shared_mem_size = (elements_per_group + 2) * sizeof(float);
    
    dim3 grid(num_groups, batch_size);
    
    group_norm_kernel<<<grid, threads_per_block, shared_mem_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        batch_size,
        num_channels,
        height,
        width,
        num_groups,
        eps
    );
}
"""

# --- C++ Logic (Interface/Bindings) ---
cpp_source = r"""
#include <torch/extension.h>

void fused_scale_pool_clamp_op(
    const torch::Tensor input,
    torch::Tensor output,
    const float scale,
    const int pool_kernel_size,
    const int pool_stride,
    const int pool_padding,
    const float clamp_min,
    const float clamp_max
);

void conv2d_op(
    const torch::Tensor input,
    const torch::Tensor weight,
    const torch::Tensor bias,
    torch::Tensor output,
    const int stride,
    const int padding,
    const int dilation
);

void group_norm_op(
    const torch::Tensor input,
    torch::Tensor output,
    const torch::Tensor weight,
    const torch::Tensor bias,
    const int num_groups,
    const float eps
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_scale_pool_clamp", &fused_scale_pool_clamp_op, "Fused scale, pool, and clamp operation");
    m.def("conv2d", &conv2d_op, "Custom Conv2D operation");
    m.def("group_norm", &group_norm_op, "Custom GroupNorm operation");
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

# --- Python Interface ---

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

# Global variables for weights and parameters
conv_weight = None
conv_bias = None
group_norm_weight = None
group_norm_bias = None

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, num_groups, scale_shape, maxpool_kernel_size, clamp_min, clamp_max]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def functional_model(
    x,
    *,
    conv_weight_param,
    conv_bias_param,
    conv_stride,
    conv_padding,
    conv_dilation,
    conv_groups,
    group_norm_weight_param,
    group_norm_bias_param,
    group_norm_num_groups,
    group_norm_eps,
    maxpool_kernel_size_param,
    maxpool_stride,
    maxpool_padding,
    maxpool_dilation,
    maxpool_ceil_mode,
    maxpool_return_indices,
    scale,
    clamp_min_param,
    clamp_max_param,
):
    global conv_weight, conv_bias, group_norm_weight, group_norm_bias
    
    # Store parameters
    conv_weight = conv_weight_param
    conv_bias = conv_bias_param
    group_norm_weight = group_norm_weight_param
    group_norm_bias = group_norm_bias_param
    
    # Calculate output dimensions for conv
    conv_out_height = (height + 2 * conv_padding - conv_dilation * (kernel_size - 1) - 1) // conv_stride + 1
    conv_out_width = (width + 2 * conv_padding - conv_dilation * (kernel_size - 1) - 1) // conv_stride + 1
    
    # Create output tensor for conv
    conv_out = torch.empty(batch_size, out_channels, conv_out_height, conv_out_width, device=x.device, dtype=x.dtype)
    
    # Perform conv2d
    fused_ext.conv2d(x, conv_weight, conv_bias, conv_out, conv_stride, conv_padding, conv_dilation)
    
    # Calculate output dimensions for group norm (same as conv output)
    gn_out_height = conv_out_height
    gn_out_width = conv_out_width
    
    # Create output tensor for group norm
    gn_out = torch.empty_like(conv_out)
    
    # Perform group norm
    fused_ext.group_norm(conv_out, gn_out, group_norm_weight, group_norm_bias, group_norm_num_groups, group_norm_eps)
    
    # Calculate output dimensions for fused operation
    pool_out_height = (gn_out_height + 2 * maxpool_padding - maxpool_kernel_size_param) // maxpool_stride + 1
    pool_out_width = (gn_out_width + 2 * maxpool_padding - maxpool_kernel_size_param) // maxpool_stride + 1
    
    # Create output tensor for fused operation
    fused_out = torch.empty(batch_size, out_channels, pool_out_height, pool_out_width, device=x.device, dtype=x.dtype)
    
    # Perform fused scale + max pool + clamp
    fused_ext.fused_scale_pool_clamp(
        gn_out, 
        fused_out, 
        scale.item() if isinstance(scale, torch.Tensor) else scale,
        maxpool_kernel_size_param,
        maxpool_stride,
        maxpool_padding,
        clamp_min_param,
        clamp_max_param
    )
    
    return fused_out
