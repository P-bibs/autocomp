# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_060810/code_5.py
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

# CUDA kernel for fused operations
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

__global__ void fused_conv3d_maxpool3d_avgpool3d_bias_sum_kernel(
    const float* __restrict__ input,
    const float* __restrict__ conv_weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ bias_tensor,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int input_depth,
    const int input_height,
    const int input_width,
    const int kernel_d,
    const int kernel_h,
    const int kernel_w,
    const int stride_d,
    const int stride_h,
    const int stride_w,
    const int padding_d,
    const int padding_h,
    const int padding_w,
    const int dilation_d,
    const int dilation_h,
    const int dilation_w,
    const int groups,
    const int pool_kernel_d,
    const int pool_kernel_h,
    const int pool_kernel_w,
    const int pool_stride_d,
    const int pool_stride_h,
    const int pool_stride_w,
    const int pool_padding_d,
    const int pool_padding_h,
    const int pool_padding_w,
    const int output_depth,
    const int output_height,
    const int output_width,
    const int pooled_depth,
    const int pooled_height,
    const int pooled_width,
    const int global_avg_d,
    const int global_avg_h,
    const int global_avg_w,
    const float divisor
) {
    // Calculate global thread index
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Early exit if thread index exceeds total number of output elements
    int total_output_elements = batch_size * out_channels * global_avg_d * global_avg_h * global_avg_w;
    if (out_idx >= total_output_elements) return;
    
    // Decompose output index
    int n = out_idx / (out_channels * global_avg_d * global_avg_h * global_avg_w);
    int c_out = (out_idx / (global_avg_d * global_avg_h * global_avg_w)) % out_channels;
    int g_idx = c_out / (out_channels / groups); // Group index
    
    int global_avg_id = (out_idx / (global_avg_h * global_avg_w)) % global_avg_d;
    int global_avg_ih = (out_idx / global_avg_w) % global_avg_h;
    int global_avg_iw = out_idx % global_avg_w;
    
    // Step 1: Conv3D + Division
    float conv_result = 0.0f;
    
    // Calculate receptive field in pooled space for this global average output
    int start_d = global_avg_id * (pooled_depth / global_avg_d);
    int end_d = start_d + (pooled_depth / global_avg_d);
    int start_h = global_avg_ih * (pooled_height / global_avg_h);
    int end_h = start_h + (pooled_height / global_avg_h);
    int start_w = global_avg_iw * (pooled_width / global_avg_w);
    int end_w = start_w + (pooled_width / global_avg_w);
    
    // Iterate over the spatial region for adaptive average pooling
    int count = 0;
    for (int pd = start_d; pd < end_d; pd++) {
        for (int ph = start_h; ph < end_h; ph++) {
            for (int pw = start_w; pw < end_w; pw++) {
                count++;
                
                // Step 2: MaxPool3D
                float max_val = -FLT_MAX;
                
                // Calculate receptive field in convolution output for this pool element
                int conv_out_start_d = pd * pool_stride_d - pool_padding_d;
                int conv_out_end_d = min(conv_out_start_d + (pool_kernel_d - 1) * 1 + 1, output_depth);
                int conv_out_start_h = ph * pool_stride_h - pool_padding_h;
                int conv_out_end_h = min(conv_out_start_h + (pool_kernel_h - 1) * 1 + 1, output_height);
                int conv_out_start_w = pw * pool_stride_w - pool_padding_w;
                int conv_out_end_w = min(conv_out_start_w + (pool_kernel_w - 1) * 1 + 1, output_width);
                
                for (int kd = 0; kd < kernel_d; kd++) {
                    for (int kh = 0; kh < kernel_h; kh++) {
                        for (int kw = 0; kw < kernel_w; kw++) {
                            // Calculate input coordinates
                            int id = conv_out_start_d * stride_d + kd * dilation_d - padding_d;
                            int ih = conv_out_start_h * stride_h + kh * dilation_h - padding_h;
                            int iw = conv_out_start_w * stride_w + kw * dilation_w - padding_w;
                            
                            // Check if in bounds
                            if (id >= 0 && id < input_depth &&
                                ih >= 0 && ih < input_height &&
                                iw >= 0 && iw < input_width) {
                                // Perform convolution
                                float sum = conv_bias[c_out];
                                int in_c_per_group = in_channels / groups;
                                
                                for (int c_in_idx = 0; c_in_idx < in_c_per_group; c_in_idx++) {
                                    int c_in = g_idx * in_c_per_group + c_in_idx;
                                    
                                    int input_idx = n * (in_channels * input_depth * input_height * input_width) +
                                                    c_in * (input_depth * input_height * input_width) +
                                                    id * (input_height * input_width) +
                                                    ih * input_width +
                                                    iw;
                                                    
                                    int weight_idx = c_out * (in_channels / groups * kernel_d * kernel_h * kernel_w) +
                                                     c_in_idx * (kernel_d * kernel_h * kernel_w) +
                                                     kd * (kernel_h * kernel_w) +
                                                     kh * kernel_w +
                                                     kw;
                                                     
                                    sum += input[input_idx] * conv_weight[weight_idx];
                                }
                                
                                // Apply division
                                float divided = sum / divisor;
                                
                                if (divided > max_val) {
                                    max_val = divided;
                                }
                            }
                        }
                    }
                }
                
                conv_result += max_val;
            }
        }
    }
    
    // Step 3: Adaptive Average Pooling (already done in loop above)
    conv_result /= count;
    
    // Step 4: Add bias
    conv_result += bias_tensor[c_out];
    
    // Step 5: Write result
    output[out_idx] = conv_result;
}

void fused_conv3d_maxpool3d_avgpool3d_bias_sum(
    const torch::Tensor input,
    const torch::Tensor conv_weight,
    const torch::Tensor conv_bias,
    const int stride_d,
    const int stride_h,
    const int stride_w,
    const int padding_d,
    const int padding_h,
    const int padding_w,
    const int dilation_d,
    const int dilation_h,
    const int dilation_w,
    const int groups,
    const int pool_kernel_d,
    const int pool_kernel_h,
    const int pool_kernel_w,
    const int pool_stride_d,
    const int pool_stride_h,
    const int pool_stride_w,
    const int pool_padding_d,
    const int pool_padding_h,
    const int pool_padding_w,
    const int global_avg_d,
    const int global_avg_h,
    const int global_avg_w,
    const float divisor,
    const torch::Tensor bias_tensor,
    torch::Tensor output
) {
    // Get tensor dimensions
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto input_depth = input.size(2);
    auto input_height = input.size(3);
    auto input_width = input.size(4);
    
    auto out_channels = conv_weight.size(0);
    auto kernel_d = conv_weight.size(2);
    auto kernel_h = conv_weight.size(3);
    auto kernel_w = conv_weight.size(4);
    
    // Calculate intermediate output sizes
    int output_depth = (input_depth + 2 * padding_d - dilation_d * (kernel_d - 1) - 1) / stride_d + 1;
    int output_height = (input_height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    int output_width = (input_width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
    
    int pooled_depth = (output_depth + 2 * pool_padding_d - pool_kernel_d) / pool_stride_d + 1;
    int pooled_height = (output_height + 2 * pool_padding_h - pool_kernel_h) / pool_stride_h + 1;
    int pooled_width = (output_width + 2 * pool_padding_w - pool_kernel_w) / pool_stride_w + 1;
    
    // Set up CUDA stream and launch parameters
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    const int threads_per_block = 256;
    const int total_output_elements = batch_size * out_channels * global_avg_d * global_avg_h * global_avg_w;
    const int num_blocks = (total_output_elements + threads_per_block - 1) / threads_per_block;
    
    // Launch kernel
    fused_conv3d_maxpool3d_avgpool3d_bias_sum_kernel<<<num_blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        conv_weight.data_ptr<float>(),
        conv_bias.data_ptr<float>(),
        bias_tensor.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_depth,
        input_height,
        input_width,
        kernel_d,
        kernel_h,
        kernel_w,
        stride_d,
        stride_h,
        stride_w,
        padding_d,
        padding_h,
        padding_w,
        dilation_d,
        dilation_h,
        dilation_w,
        groups,
        pool_kernel_d,
        pool_kernel_h,
        pool_kernel_w,
        pool_stride_d,
        pool_stride_h,
        pool_stride_w,
        pool_padding_d,
        pool_padding_h,
        pool_padding_w,
        output_depth,
        output_height,
        output_width,
        pooled_depth,
        pooled_height,
        pooled_width,
        global_avg_d,
        global_avg_h,
        global_avg_w,
        divisor
    );
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("CUDA kernel launch failed: ", cudaGetErrorString(err));
    }
}
"""

# C++ source for PyTorch binding
cpp_source = r"""
#include <torch/extension.h>

void fused_conv3d_maxpool3d_avgpool3d_bias_sum(
    const torch::Tensor input,
    const torch::Tensor conv_weight,
    const torch::Tensor conv_bias,
    const int stride_d,
    const int stride_h,
    const int stride_w,
    const int padding_d,
    const int padding_h,
    const int padding_w,
    const int dilation_d,
    const int dilation_h,
    const int dilation_w,
    const int groups,
    const int pool_kernel_d,
    const int pool_kernel_h,
    const int pool_kernel_w,
    const int pool_stride_d,
    const int pool_stride_h,
    const int pool_stride_w,
    const int pool_padding_d,
    const int pool_padding_h,
    const int pool_padding_w,
    const int global_avg_d,
    const int global_avg_h,
    const int global_avg_w,
    const float divisor,
    const torch::Tensor bias_tensor,
    torch::Tensor output
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_conv3d_maxpool3d_avgpool3d_bias_sum, "Fused Conv3D+MaxPool3D+AdaptiveAvgPool3D+Bias+Sum");
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
    # Extract stride, padding, dilation values
    conv_stride_d, conv_stride_h, conv_stride_w = conv_stride
    conv_padding_d, conv_padding_h, conv_padding_w = conv_padding
    conv_dilation_d, conv_dilation_h, conv_dilation_w = conv_dilation
    
    pool_kernel_d, pool_kernel_h, pool_kernel_w = max_pool_kernel_size
    pool_stride_d, pool_stride_h, pool_stride_w = max_pool_stride
    pool_padding_d, pool_padding_h, pool_padding_w = max_pool_padding
    
    global_avg_d, global_avg_h, global_avg_w = global_avg_pool_output_size
    
    # Create output tensor
    batch_size = x.size(0)
    out_channels = conv_weight.size(0)
    output = torch.empty(
        batch_size, out_channels, global_avg_d, global_avg_h, global_avg_w,
        dtype=x.dtype, device=x.device
    )
    
    # Call fused kernel
    fused_ext.fused_op(
        x.contiguous(),
        conv_weight.contiguous(),
        conv_bias.contiguous(),
        conv_stride_d,
        conv_stride_h,
        conv_stride_w,
        conv_padding_d,
        conv_padding_h,
        conv_padding_w,
        conv_dilation_d,
        conv_dilation_h,
        conv_dilation_w,
        conv_groups,
        pool_kernel_d,
        pool_kernel_h,
        pool_kernel_w,
        pool_stride_d,
        pool_stride_h,
        pool_stride_w,
        pool_padding_d,
        pool_padding_h,
        pool_padding_w,
        global_avg_d,
        global_avg_h,
        global_avg_w,
        divisor,
        bias.contiguous(),
        output
    )
    
    # Apply sum reduction
    output = torch.sum(output, dim=sum_dim)
    
    return output

batch_size   = 128  
in_channels  = 8            
out_channels = 16  
depth = 16; height = width = 64 
kernel_size = (3, 3, 3)
divisor = 2.0
pool_size = (2, 2, 2)
bias_shape = (out_channels, 1, 1, 1)
sum_dim = 1

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim]

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]
