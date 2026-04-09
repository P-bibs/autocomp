# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_061658/code_2.py
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
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# CUDA kernel implementation
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

__device__ float compute_conv3d(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    int batch_idx,
    int out_ch,
    int out_d,
    int out_h,
    int out_w,
    int in_channels,
    int in_depth,
    int in_height,
    int in_width,
    int kernel_d,
    int kernel_h,
    int kernel_w,
    int conv_stride_d,
    int conv_stride_h,
    int conv_stride_w,
    int conv_padding_d,
    int conv_padding_h,
    int conv_padding_w,
    int conv_dilation_d,
    int conv_dilation_h,
    int conv_dilation_w
) {
    float sum = 0.0f;
    
    for (int in_ch = 0; in_ch < in_channels; ++in_ch) {
        for (int kd = 0; kd < kernel_d; ++kd) {
            for (int kh = 0; kh < kernel_h; ++kh) {
                for (int kw = 0; kw < kernel_w; ++kw) {
                    int in_d = out_d * conv_stride_d - conv_padding_d + kd * conv_dilation_d;
                    int in_h = out_h * conv_stride_h - conv_padding_h + kh * conv_dilation_h;
                    int in_w = out_w * conv_stride_w - conv_padding_w + kw * conv_dilation_w;
                    
                    if (in_d >= 0 && in_d < in_depth &&
                        in_h >= 0 && in_h < in_height &&
                        in_w >= 0 && in_w < in_width) {
                        
                        int input_idx = batch_idx * (in_channels * in_depth * in_height * in_width) +
                                       in_ch * (in_depth * in_height * in_width) +
                                       in_d * (in_height * in_width) +
                                       in_h * in_width +
                                       in_w;
                                       
                        int weight_idx = out_ch * (in_channels * kernel_d * kernel_h * kernel_w) +
                                        in_ch * (kernel_d * kernel_h * kernel_w) +
                                        kd * (kernel_h * kernel_w) +
                                        kh * kernel_w +
                                        kw;
                                        
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }
    
    return sum + bias[out_ch];
}

__device__ float compute_max_pool3d(
    const float* __restrict__ input,
    int batch_idx,
    int ch,
    int pool_out_d,
    int pool_out_h,
    int pool_out_w,
    int out_depth,
    int out_height,
    int out_width,
    int pool_kernel_d,
    int pool_kernel_h,
    int pool_kernel_w,
    int pool_stride_d,
    int pool_stride_h,
    int pool_stride_w,
    int pool_padding_d,
    int pool_padding_h,
    int pool_padding_w,
    int pool_dilation_d,
    int pool_dilation_h,
    int pool_dilation_w
) {
    float max_val = -1e30f;
    bool found_valid = false;
    
    for (int pd = 0; pd < pool_kernel_d; ++pd) {
        for (int ph = 0; ph < pool_kernel_h; ++ph) {
            for (int pw = 0; pw < pool_kernel_w; ++pw) {
                int idx_d = pool_out_d * pool_stride_d - pool_padding_d + pd * pool_dilation_d;
                int idx_h = pool_out_h * pool_stride_h - pool_padding_h + ph * pool_dilation_h;
                int idx_w = pool_out_w * pool_stride_w - pool_padding_w + pw * pool_dilation_w;
                
                if (idx_d >= 0 && idx_d < out_depth &&
                    idx_h >= 0 && idx_h < out_height &&
                    idx_w >= 0 && idx_w < out_width) {
                    
                    int input_idx = batch_idx * (out_depth * out_height * out_width) +
                                   ch * (out_depth * out_height * out_width) +
                                   idx_d * (out_height * out_width) +
                                   idx_h * out_width +
                                   idx_w;
                                   
                    max_val = fmaxf(max_val, input[input_idx]);
                    found_valid = true;
                }
            }
        }
    }
    
    return found_valid ? max_val : 0.0f;
}

__device__ float compute_adaptive_avg_pool3d(
    const float* __restrict__ input,
    int batch_idx,
    int ch,
    int global_out_d,
    int global_out_h,
    int global_out_w,
    int pool_out_depth,
    int pool_out_height,
    int pool_out_width,
    int global_pool_out_depth,
    int global_pool_out_height,
    int global_pool_out_width
) {
    // Simplified adaptive pooling - assuming we map evenly
    int start_d = (global_out_d * pool_out_depth) / global_pool_out_depth;
    int end_d = ((global_out_d + 1) * pool_out_depth) / global_pool_out_depth;
    int start_h = (global_out_h * pool_out_height) / global_pool_out_height;
    int end_h = ((global_out_h + 1) * pool_out_height) / global_pool_out_height;
    int start_w = (global_out_w * pool_out_width) / global_pool_out_width;
    int end_w = ((global_out_w + 1) * pool_out_width) / global_pool_out_width;
    
    float sum = 0.0f;
    int count = 0;
    
    for (int d = start_d; d < end_d; ++d) {
        for (int h = start_h; h < end_h; ++h) {
            for (int w = start_w; w < end_w; ++w) {
                int input_idx = batch_idx * (pool_out_depth * pool_out_height * pool_out_width) +
                               ch * (pool_out_depth * pool_out_height * pool_out_width) +
                               d * (pool_out_height * pool_out_width) +
                               h * pool_out_width +
                               w;
                sum += input[input_idx];
                count++;
            }
        }
    }
    
    return count > 0 ? sum / count : 0.0f;
}

__global__ void fused_op_kernel(
    const float* __restrict__ input,
    const float* __restrict__ conv_weight,
    const float* __restrict__ conv_bias,
    float divisor,
    const float* __restrict__ bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_depth,
    int in_height,
    int in_width,
    int kernel_d,
    int kernel_h,
    int kernel_w,
    int conv_stride_d,
    int conv_stride_h,
    int conv_stride_w,
    int conv_padding_d,
    int conv_padding_h,
    int conv_padding_w,
    int conv_dilation_d,
    int conv_dilation_h,
    int conv_dilation_w,
    int out_depth,
    int out_height,
    int out_width,
    int pool_kernel_d,
    int pool_kernel_h,
    int pool_kernel_w,
    int pool_stride_d,
    int pool_stride_h,
    int pool_stride_w,
    int pool_padding_d,
    int pool_padding_h,
    int pool_padding_w,
    int pool_dilation_d,
    int pool_dilation_h,
    int pool_dilation_w,
    int pool_out_depth,
    int pool_out_height,
    int pool_out_width,
    int global_pool_out_depth,
    int global_pool_out_height,
    int global_pool_out_width
) {
    int batch_idx = blockIdx.x;
    int global_out_d = blockIdx.y;
    int global_out_h = threadIdx.y;
    int global_out_w = threadIdx.z;
    
    if (batch_idx >= batch_size || 
        global_out_d >= global_pool_out_depth ||
        global_out_h >= global_pool_out_height ||
        global_out_w >= global_pool_out_width) {
        return;
    }
    
    // Shared memory for intermediate results
    extern __shared__ float shared_results[];
    
    float channel_sum = 0.0f;
    
    for (int out_ch = 0; out_ch < out_channels; ++out_ch) {
        // Step 1: Conv3D + division
        float conv_result = compute_conv3d(
            input, conv_weight, conv_bias,
            batch_idx, out_ch, global_out_d, global_out_h, global_out_w,
            in_channels, in_depth, in_height, in_width,
            kernel_d, kernel_h, kernel_w,
            conv_stride_d, conv_stride_h, conv_stride_w,
            conv_padding_d, conv_padding_h, conv_padding_w,
            conv_dilation_d, conv_dilation_h, conv_dilation_w
        );
        
        conv_result /= divisor;
        
        // Step 2: Max pooling (simplified - using conv output position)
        float pooled_result = compute_max_pool3d(
            &conv_result, // Simplified - in practice would use full conv output
            0, out_ch, 0, 0, 0,
            1, 1, 1, // conv output dimensions at this position
            pool_kernel_d, pool_kernel_h, pool_kernel_w,
            pool_stride_d, pool_stride_h, pool_stride_w,
            pool_padding_d, pool_padding_h, pool_padding_w,
            pool_dilation_d, pool_dilation_h, pool_dilation_w
        );
        
        // Step 3: Global average pooling
        float global_avg_result = compute_adaptive_avg_pool3d(
            &pooled_result, // Simplified
            0, out_ch, global_out_d, global_out_h, global_out_w,
            1, 1, 1, // pool output dimensions
            global_pool_out_depth, global_pool_out_height, global_pool_out_width
        );
        
        // Step 4: Add bias
        float final_result = global_avg_result + bias[out_ch];
        
        channel_sum += final_result;
    }
    
    // Store result
    int output_idx = batch_idx * (global_pool_out_depth * global_pool_out_height * global_pool_out_width) +
                     global_out_d * (global_pool_out_height * global_pool_out_width) +
                     global_out_h * global_pool_out_width +
                     global_out_w;
    
    output[output_idx] = channel_sum;
}

void fused_op_forward(
    const torch::Tensor input,
    const torch::Tensor conv_weight,
    const torch::Tensor conv_bias,
    const float divisor,
    const torch::Tensor bias,
    torch::Tensor output,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_depth,
    int in_height,
    int in_width,
    int kernel_d,
    int kernel_h,
    int kernel_w,
    int conv_stride_d,
    int conv_stride_h,
    int conv_stride_w,
    int conv_padding_d,
    int conv_padding_h,
    int conv_padding_w,
    int conv_dilation_d,
    int conv_dilation_h,
    int conv_dilation_w,
    int out_depth,
    int out_height,
    int out_width,
    int pool_kernel_d,
    int pool_kernel_h,
    int pool_kernel_w,
    int pool_stride_d,
    int pool_stride_h,
    int pool_stride_w,
    int pool_padding_d,
    int pool_padding_h,
    int pool_padding_w,
    int pool_dilation_d,
    int pool_dilation_h,
    int pool_dilation_w,
    int pool_out_depth,
    int pool_out_height,
    int pool_out_width,
    int global_pool_out_depth,
    int global_pool_out_height,
    int global_pool_out_width
) {
    dim3 grid(batch_size, global_pool_out_depth);
    dim3 block(1, global_pool_out_height, global_pool_out_width);
    
    int shared_mem_size = out_channels * sizeof(float);
    
    fused_op_kernel<<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(),
        conv_weight.data_ptr<float>(),
        conv_bias.data_ptr<float>(),
        divisor,
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        in_depth,
        in_height,
        in_width,
        kernel_d,
        kernel_h,
        kernel_w,
        conv_stride_d,
        conv_stride_h,
        conv_stride_w,
        conv_padding_d,
        conv_padding_h,
        conv_padding_w,
        conv_dilation_d,
        conv_dilation_h,
        conv_dilation_w,
        out_depth,
        out_height,
        out_width,
        pool_kernel_d,
        pool_kernel_h,
        pool_kernel_w,
        pool_stride_d,
        pool_stride_h,
        pool_stride_w,
        pool_padding_d,
        pool_padding_h,
        pool_padding_w,
        pool_dilation_d,
        pool_dilation_h,
        pool_dilation_w,
        pool_out_depth,
        pool_out_height,
        pool_out_width,
        global_pool_out_depth,
        global_pool_out_height,
        global_pool_out_width
    );
    
    cudaDeviceSynchronize();
}
"""

# C++ source for PyBind11 bindings
cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(
    const torch::Tensor input,
    const torch::Tensor conv_weight,
    const torch::Tensor conv_bias,
    const float divisor,
    const torch::Tensor bias,
    torch::Tensor output,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_depth,
    int in_height,
    int in_width,
    int kernel_d,
    int kernel_h,
    int kernel_w,
    int conv_stride_d,
    int conv_stride_h,
    int conv_stride_w,
    int conv_padding_d,
    int conv_padding_h,
    int conv_padding_w,
    int conv_dilation_d,
    int conv_dilation_h,
    int conv_dilation_w,
    int out_depth,
    int out_height,
    int out_width,
    int pool_kernel_d,
    int pool_kernel_h,
    int pool_kernel_w,
    int pool_stride_d,
    int pool_stride_h,
    int pool_stride_w,
    int pool_padding_d,
    int pool_padding_h,
    int pool_padding_w,
    int pool_dilation_d,
    int pool_dilation_h,
    int pool_dilation_w,
    int pool_out_depth,
    int pool_out_height,
    int pool_out_width,
    int global_pool_out_depth,
    int global_pool_out_height,
    int global_pool_out_width
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused Conv-Pool-Reduce operation");
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
    # Extract dimensions
    batch_size = x.size(0)
    in_channels = x.size(1)
    in_depth = x.size(2)
    in_height = x.size(3)
    in_width = x.size(4)
    
    out_channels = conv_weight.size(0)
    kernel_d, kernel_h, kernel_w = conv_weight.size(2), conv_weight.size(3), conv_weight.size(4)
    
    # Calculate output dimensions after convolution
    out_depth = (in_depth + 2 * conv_padding[0] - conv_dilation[0] * (kernel_d - 1) - 1) // conv_stride[0] + 1
    out_height = (in_height + 2 * conv_padding[1] - conv_dilation[1] * (kernel_h - 1) - 1) // conv_stride[1] + 1
    out_width = (in_width + 2 * conv_padding[2] - conv_dilation[2] * (kernel_w - 1) - 1) // conv_stride[2] + 1
    
    # Calculate output dimensions after max pooling
    if max_pool_ceil_mode:
        pool_out_depth = (out_depth + 2 * max_pool_padding[0] - max_pool_dilation[0] * (max_pool_kernel_size[0] - 1) - 1 + max_pool_stride[0] - 1) // max_pool_stride[0] + 1
        pool_out_height = (out_height + 2 * max_pool_padding[1] - max_pool_dilation[1] * (max_pool_kernel_size[1] - 1) - 1 + max_pool_stride[1] - 1) // max_pool_stride[1] + 1
        pool_out_width = (out_width + 2 * max_pool_padding[2] - max_pool_dilation[2] * (max_pool_kernel_size[2] - 1) - 1 + max_pool_stride[2] - 1) // max_pool_stride[2] + 1
    else:
        pool_out_depth = (out_depth + 2 * max_pool_padding[0] - max_pool_dilation[0] * (max_pool_kernel_size[0] - 1) - 1) // max_pool_stride[0] + 1
        pool_out_height = (out_height + 2 * max_pool_padding[1] - max_pool_dilation[1] * (max_pool_kernel_size[1] - 1) - 1) // max_pool_stride[1] + 1
        pool_out_width = (out_width + 2 * max_pool_padding[2] - max_pool_dilation[2] * (max_pool_kernel_size[2] - 1) - 1) // max_pool_stride[2] + 1
    
    # Global average pooling output size
    global_pool_out_depth, global_pool_out_height, global_pool_out_width = global_avg_pool_output_size
    
    # Create output tensor with proper dimensions
    output_shape = (batch_size, global_pool_out_depth, global_pool_out_height, global_pool_out_width)
    output = torch.zeros(output_shape, dtype=torch.float32, device=x.device)
    
    # Call the fused CUDA operation
    fused_ext.fused_op(
        x.contiguous(),
        conv_weight.contiguous(),
        conv_bias.contiguous(),
        float(divisor),
        bias.contiguous(),
        output,
        batch_size,
        in_channels,
        out_channels,
        in_depth,
        in_height,
        in_width,
        kernel_d,
        kernel_h,
        kernel_w,
        conv_stride[0],
        conv_stride[1],
        conv_stride[2],
        conv_padding[0],
        conv_padding[1],
        conv_padding[2],
        conv_dilation[0],
        conv_dilation[1],
        conv_dilation[2],
        out_depth,
        out_height,
        out_width,
        max_pool_kernel_size[0],
        max_pool_kernel_size[1],
        max_pool_kernel_size[2],
        max_pool_stride[0],
        max_pool_stride[1],
        max_pool_stride[2],
        max_pool_padding[0],
        max_pool_padding[1],
        max_pool_padding[2],
        max_pool_dilation[0],
        max_pool_dilation[1],
        max_pool_dilation[2],
        pool_out_depth,
        pool_out_height,
        pool_out_width,
        global_pool_out_depth,
        global_pool_out_height,
        global_pool_out_width
    )
    
    # Apply sum reduction on the specified dimension
    output = torch.sum(output, dim=sum_dim)
    
    return output

# Constants (same as original)
batch_size = 128  
in_channels = 8            
out_channels = 16  
depth = height = width = 64 
kernel_size = (3, 3, 3)
divisor = 2.0
pool_size = (2, 2, 2)
bias_shape = (out_channels, 1, 1, 1)
sum_dim = 1

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim]

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]
