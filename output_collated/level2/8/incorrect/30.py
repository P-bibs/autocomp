# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_060810/code_1.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'divisor', 'pool_size', 'bias_shape', 'sum_dim']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_weight', 'conv_bias', 'conv_stride', 'conv_padding', 'conv_dilation', 'conv_groups', 'max_pool_kernel_size', 'max_pool_stride', 'max_pool_padding', 'max_pool_dilation', 'max_pool_ceil_mode', 'max_pool_return_indices', 'global_avg_pool_output_size', 'divisor', 'bias', 'sum_dim']
REQUIRED_FLAT_STATE_NAMES = ['conv_weight', 'conv_bias', 'bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a 3D convolution, divides by a constant, applies max pooling,
    global average pooling, adds a bias term, and sums along a specific dimension.
    """

    def __init__(self, in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.divisor = divisor
        self.max_pool = nn.MaxPool3d(pool_size)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.sum_dim = sum_dim

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
    # State for conv (nn.Conv3d)
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
    # State for max_pool (nn.MaxPool3d)
    state_kwargs['max_pool_kernel_size'] = model.max_pool.kernel_size
    state_kwargs['max_pool_stride'] = model.max_pool.stride
    state_kwargs['max_pool_padding'] = model.max_pool.padding
    state_kwargs['max_pool_dilation'] = model.max_pool.dilation
    state_kwargs['max_pool_ceil_mode'] = model.max_pool.ceil_mode
    state_kwargs['max_pool_return_indices'] = model.max_pool.return_indices
    # State for global_avg_pool (nn.AdaptiveAvgPool3d)
    state_kwargs['global_avg_pool_output_size'] = model.global_avg_pool.output_size
    if 'divisor' in flat_state:
        state_kwargs['divisor'] = flat_state['divisor']
    else:
        state_kwargs['divisor'] = getattr(model, 'divisor')
    if 'bias' in flat_state:
        state_kwargs['bias'] = flat_state['bias']
    else:
        state_kwargs['bias'] = getattr(model, 'bias')
    if 'sum_dim' in flat_state:
        state_kwargs['sum_dim'] = flat_state['sum_dim']
    else:
        state_kwargs['sum_dim'] = getattr(model, 'sum_dim')
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

# Define the fused CUDA kernel
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define CUDA_1D_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

__device__ inline int get_idx_5d(int d0, int d1, int d2, int d3, int d4,
                                 int s1, int s2, int s3, int s4) {
    return d0 * s1 * s2 * s3 * s4 +
           d1 * s2 * s3 * s4 +
           d2 * s3 * s4 +
           d3 * s4 +
           d4;
}

__global__ void fused_conv_pool_reduce_kernel(
    const float* __restrict__ input,
    const float* __restrict__ conv_weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_depth, int in_height, int in_width,
    int out_depth, int out_height, int out_width,
    int conv_kd, int conv_kh, int conv_kw,
    int conv_stride_d, int conv_stride_h, int conv_stride_w,
    int conv_pad_d, int conv_pad_h, int conv_pad_w,
    int pool_kd, int pool_kh, int pool_kw,
    int pool_stride_d, int pool_stride_h, int pool_stride_w,
    int pool_pad_d, int pool_pad_h, int pool_pad_w,
    float divisor,
    int sum_dim,
    int global_avg_pool_d, int global_avg_pool_h, int global_avg_pool_w
) {
    // Calculate global thread index for output spatial dimensions
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = batch_size * out_channels * out_depth * out_height * out_width;
    
    if (tid >= total_threads) return;
    
    // Decode thread index to get batch, channel, spatial coordinates
    int batch_idx = tid / (out_channels * out_depth * out_height * out_width);
    int temp = tid % (out_channels * out_depth * out_height * out_width);
    int out_c = temp / (out_depth * out_height * out_width);
    temp = temp % (out_depth * out_height * out_width);
    int out_d = temp / (out_height * out_width);
    temp = temp % (out_height * out_width);
    int out_h = temp / out_width;
    int out_w = temp % out_width;
    
    // Step 1: Conv3D operation
    float conv_result = 0.0f;
    if (conv_bias != nullptr) {
        conv_result = conv_bias[out_c];
    }
    
    for (int in_c = 0; in_c < in_channels; ++in_c) {
        for (int kd = 0; kd < conv_kd; ++kd) {
            for (int kh = 0; kh < conv_kh; ++kh) {
                for (int kw = 0; kw < conv_kw; ++kw) {
                    int in_d = out_d * conv_stride_d - conv_pad_d + kd;
                    int in_h = out_h * conv_stride_h - conv_pad_h + kh;
                    int in_w = out_w * conv_stride_w - conv_pad_w + kw;
                    
                    if (in_d >= 0 && in_d < in_depth &&
                        in_h >= 0 && in_h < in_height &&
                        in_w >= 0 && in_w < in_width) {
                        
                        int input_idx = get_idx_5d(batch_idx, in_c, in_d, in_h, in_w,
                                                   in_channels, in_depth, in_height, in_width);
                        
                        int weight_idx = get_idx_5d(out_c, in_c, kd, kh, kw,
                                                    in_channels, conv_kd, conv_kh, conv_kw);
                        
                        conv_result += input[input_idx] * conv_weight[weight_idx];
                    }
                }
            }
        }
    }
    
    // Step 2: Division
    float div_result = conv_result / divisor;
    
    // Step 3: MaxPool3D operation
    float pool_result = -1e38f; // Negative infinity
    bool has_valid = false;
    
    // For this simplified implementation, we'll do 1x1x1 pooling when at the final spatial location
    // which mimics adaptive average pooling to a single value
    if (global_avg_pool_d == 1 && global_avg_pool_h == 1 && global_avg_pool_w == 1) {
        // Only pool at the center of the global average pool region
        int pd_center_d = pool_kd / 2;
        int ph_center_h = pool_kh / 2;
        int pw_center_w = pool_kw / 2;
        
        int pool_in_d = out_d * pool_stride_d - pool_pad_d + pd_center_d;
        int pool_in_h = out_h * pool_stride_h - pool_pad_h + ph_center_h;
        int pool_in_w = out_w * pool_stride_w - pool_pad_w + pw_center_w;
        
        if (pool_in_d >= 0 && pool_in_d < out_depth &&
            pool_in_h >= 0 && pool_in_h < out_height &&
            pool_in_w >= 0 && pool_in_w < out_width) {
            has_valid = true;
            pool_result = fmaxf(pool_result, div_result);
        }
    } else {
        // Normal max pooling
        for (int pd = 0; pd < pool_kd; ++pd) {
            for (int ph = 0; ph < pool_kh; ++ph) {
                for (int pw = 0; pw < pool_kw; ++pw) {
                    int pool_in_d = out_d * pool_stride_d - pool_pad_d + pd;
                    int pool_in_h = out_h * pool_stride_h - pool_pad_h + ph;
                    int pool_in_w = out_w * pool_stride_w - pool_pad_w + pw;
                    
                    if (pool_in_d >= 0 && pool_in_d < out_depth &&
                        pool_in_h >= 0 && pool_in_h < out_height &&
                        pool_in_w >= 0 && pool_in_w < out_width) {
                        has_valid = true;
                        pool_result = fmaxf(pool_result, div_result);
                    }
                }
            }
        }
    }
    
    if (!has_valid) {
        pool_result = div_result;
    }
    
    // Step 4: AdaptiveAvgPool3D (simplified for when output is 1x1x1)
    float avg_result = pool_result;
    
    // If this is the last spatial element (implies global pooling result)
    if (global_avg_pool_d == 1 && global_avg_pool_h == 1 && global_avg_pool_w == 1 &&
        out_d == out_depth - 1 && out_h == out_height - 1 && out_w == out_width - 1) {
        // For simplicity, we'll average by dividing by the total number of elements
        avg_result = pool_result / (out_depth * out_height * out_width);
    }
    
    // Step 5: Add bias
    float bias_result = avg_result + bias[out_c];
    
    // Step 6: Sum reduction along sum_dim (channel dimension)
    if (sum_dim == 1) {  // Summing over channel dimension
        atomicAdd(&output[batch_idx], bias_result);
    }
}

void fused_forward_op(
    const torch::Tensor input,
    const torch::Tensor conv_weight,
    const torch::Tensor conv_bias,
    const torch::Tensor bias,
    torch::Tensor output,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_depth, int in_height, int in_width,
    int out_depth, int out_height, int out_width,
    int conv_kd, int conv_kh, int conv_kw,
    int conv_stride_d, int conv_stride_h, int conv_stride_w,
    int conv_pad_d, int conv_pad_h, int conv_pad_w,
    int pool_kd, int pool_kh, int pool_kw,
    int pool_stride_d, int pool_stride_h, int pool_stride_w,
    int pool_pad_d, int pool_pad_h, int pool_pad_w,
    float divisor,
    int sum_dim,
    int global_avg_pool_d, int global_avg_pool_h, int global_avg_pool_w
) {
    int total_threads = batch_size * out_channels * out_depth * out_height * out_width;
    int threads_per_block = 256;
    int blocks = (total_threads + threads_per_block - 1) / threads_per_block;
    
    fused_conv_pool_reduce_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        conv_weight.data_ptr<float>(),
        conv_bias.defined() ? conv_bias.data_ptr<float>() : nullptr,
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        in_depth, in_height, in_width,
        out_depth, out_height, out_width,
        conv_kd, conv_kh, conv_kw,
        conv_stride_d, conv_stride_h, conv_stride_w,
        conv_pad_d, conv_pad_h, conv_pad_w,
        pool_kd, pool_kh, pool_kw,
        pool_stride_d, pool_stride_h, pool_stride_w,
        pool_pad_d, pool_pad_h, pool_pad_w,
        divisor,
        sum_dim,
        global_avg_pool_d, global_avg_pool_h, global_avg_pool_w
    );
}
"""

# C++ binding code
cpp_source = r"""
#include <torch/extension.h>

void fused_forward_op(
    const torch::Tensor input,
    const torch::Tensor conv_weight,
    const torch::Tensor conv_bias,
    const torch::Tensor bias,
    torch::Tensor output,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_depth, int in_height, int in_width,
    int out_depth, int out_height, int out_width,
    int conv_kd, int conv_kh, int conv_kw,
    int conv_stride_d, int conv_stride_h, int conv_stride_w,
    int conv_pad_d, int conv_pad_h, int conv_pad_w,
    int pool_kd, int pool_kh, int pool_kw,
    int pool_stride_d, int pool_stride_h, int pool_stride_w,
    int pool_pad_d, int pool_pad_h, int pool_pad_w,
    float divisor,
    int sum_dim,
    int global_avg_pool_d, int global_avg_pool_h, int global_avg_pool_w
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_forward_op", &fused_forward_op, "Fused Conv+Pool+Reduce forward operation");
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
    max_pool_kernel_size,
    max_pool_stride,
    max_pool_padding,
    max_pool_dilation,
    max_pool_ceil_mode,
    max_pool_return_indices,
    global_avg_pool_output_size,
    divisor,
    bias,
    sum_dim,
):
    batch_size, in_channels, in_depth, in_height, in_width = x.shape
    out_channels = conv_weight.shape[0]
    
    # Calculate output dimensions after conv3d
    conv_kd, conv_kh, conv_kw = conv_weight.shape[2], conv_weight.shape[3], conv_weight.shape[4]
    conv_stride_d, conv_stride_h, conv_stride_w = conv_stride if isinstance(conv_stride, (tuple, list)) else (conv_stride, conv_stride, conv_stride)
    conv_pad_d, conv_pad_h, conv_pad_w = conv_padding if isinstance(conv_padding, (tuple, list)) else (conv_padding, conv_padding, conv_padding)
    conv_dilation_d, conv_dilation_h, conv_dilation_w = conv_dilation if isinstance(conv_dilation, (tuple, list)) else (conv_dilation, conv_dilation, conv_dilation)
    
    out_depth = (in_depth + 2 * conv_pad_d - conv_dilation_d * (conv_kd - 1) - 1) // conv_stride_d + 1
    out_height = (in_height + 2 * conv_pad_h - conv_dilation_h * (conv_kh - 1) - 1) // conv_stride_h + 1
    out_width = (in_width + 2 * conv_pad_w - conv_dilation_w * (conv_kw - 1) - 1) // conv_stride_w + 1
    
    # Pool kernel sizes
    pool_kd, pool_kh, pool_kw = max_pool_kernel_size if isinstance(max_pool_kernel_size, (tuple, list)) else (max_pool_kernel_size, max_pool_kernel_size, max_pool_kernel_size)
    pool_stride_d, pool_stride_h, pool_stride_w = max_pool_stride if isinstance(max_pool_stride, (tuple, list)) else (max_pool_stride, max_pool_stride, max_pool_stride)
    pool_pad_d, pool_pad_h, pool_pad_w = max_pool_padding if isinstance(max_pool_padding, (tuple, list)) else (max_pool_padding, max_pool_padding, max_pool_padding)
    
    # Global average pooling size
    global_avg_pool_d, global_avg_pool_h, global_avg_pool_w = global_avg_pool_output_size if isinstance(global_avg_pool_output_size, (tuple, list)) else (global_avg_pool_output_size, global_avg_pool_output_size, global_avg_pool_output_size)
    
    # Create output tensor
    output = torch.zeros(batch_size, dtype=x.dtype, device=x.device)
    
    # Call the fused kernel
    fused_ext.fused_forward_op(
        x.contiguous(), 
        conv_weight.contiguous(), 
        conv_bias.contiguous() if conv_bias is not None else torch.empty(0, device=x.device), 
        bias.contiguous(),
        output,
        batch_size,
        in_channels,
        out_channels,
        in_depth, in_height, in_width,
        out_depth, out_height, out_width,
        conv_kd, conv_kh, conv_kw,
        conv_stride_d, conv_stride_h, conv_stride_w,
        conv_pad_d, conv_pad_h, conv_pad_w,
        pool_kd, pool_kh, pool_kw,
        pool_stride_d, pool_stride_h, pool_stride_w,
        pool_pad_d, pool_pad_h, pool_pad_w,
        float(divisor),
        int(sum_dim),
        global_avg_pool_d, global_avg_pool_h, global_avg_pool_w
    )
    
    return output

batch_size = 128  
in_channels = 8            
out_channels = 16  
depth = height = width = 16
kernel_size = (3, 3, 3)
divisor = 2.0
pool_size = (2, 2, 2)
bias_shape = (out_channels, 1, 1, 1)
sum_dim = 1

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim]

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]
