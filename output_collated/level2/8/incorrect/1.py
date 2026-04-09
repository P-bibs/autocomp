# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_053107/code_0.py
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

# CUDA kernel code
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define THREADS_PER_BLOCK 256
#define WARP_SIZE 32

__global__ void fused_op_kernel(
    const float* __restrict__ input,
    const float* __restrict__ conv_weight,
    const float* __restrict__ conv_bias,
    const float divisor,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_depth,
    int input_height,
    int input_width,
    int kernel_d, 
    int kernel_h,
    int kernel_w,
    int conv_stride_d,
    int conv_stride_h,
    int conv_stride_w,
    int conv_pad_d,
    int conv_pad_h,
    int conv_pad_w,
    int pool_kernel_d,
    int pool_kernel_h,
    int pool_kernel_w,
    int pool_stride_d,
    int pool_stride_h,
    int pool_stride_w,
    int pool_pad_d,
    int pool_pad_h,
    int pool_pad_w,
    int global_avg_pool_d,
    int global_avg_pool_h,
    int global_avg_pool_w,
    int sum_dim
) {
    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Output dimensions after all operations
    int conv_out_d = (input_depth + 2*conv_pad_d - kernel_d) / conv_stride_d + 1;
    int conv_out_h = (input_height + 2*conv_pad_h - kernel_h) / conv_stride_h + 1;
    int conv_out_w = (input_width + 2*conv_pad_w - kernel_w) / conv_stride_w + 1;
    
    int pool_out_d = (conv_out_d + 2*pool_pad_d - pool_kernel_d) / pool_stride_d + 1;
    int pool_out_h = (conv_out_h + 2*pool_pad_h - pool_kernel_h) / pool_stride_h + 1;
    int pool_out_w = (conv_out_w + 2*pool_pad_w - pool_kernel_w) / pool_stride_w + 1;
    
    int total_output_elements = batch_size * out_channels * pool_out_d * pool_out_h * pool_out_w;
    
    if (idx >= total_output_elements) return;
    
    // Decode output coordinates after pooling
    int temp = idx;
    int pool_w_idx = temp % pool_out_w;
    temp /= pool_out_w;
    int pool_h_idx = temp % pool_out_h;
    temp /= pool_out_h;
    int pool_d_idx = temp % pool_out_d;
    temp /= pool_out_d;
    int c_idx = temp % out_channels;
    int b_idx = temp / out_channels;
    
    // Calculate corresponding coordinates in conv output
    int conv_d_start = pool_d_idx * pool_stride_d - pool_pad_d;
    int conv_h_start = pool_h_idx * pool_stride_h - pool_pad_h;
    int conv_w_start = pool_w_idx * pool_stride_w - pool_pad_w;
    
    // Max pooling operation
    float max_val = -FLT_MAX;
    for (int pd = 0; pd < pool_kernel_d; pd++) {
        for (int ph = 0; ph < pool_kernel_h; ph++) {
            for (int pw = 0; pw < pool_kernel_w; pw++) {
                int conv_d = conv_d_start + pd;
                int conv_h = conv_h_start + ph;
                int conv_w = conv_w_start + pw;
                
                if (conv_d >= 0 && conv_d < conv_out_d && 
                    conv_h >= 0 && conv_h < conv_out_h && 
                    conv_w >= 0 && conv_w < conv_out_w) {
                    
                    // Calculate conv value at this position
                    int input_d = conv_d * conv_stride_d - conv_pad_d;
                    int input_h = conv_h * conv_stride_h - conv_pad_h;
                    int input_w = conv_w * conv_stride_w - conv_pad_w;
                    
                    float conv_result = (conv_bias != nullptr) ? conv_bias[c_idx] : 0.0f;
                    
                    for (int kd = 0; kd < kernel_d; kd++) {
                        for (int kh = 0; kh < kernel_h; kh++) {
                            for (int kw = 0; kw < kernel_w; kw++) {
                                int in_d = input_d + kd;
                                int in_h = input_h + kh;
                                int in_w = input_w + kw;
                                
                                if (in_d >= 0 && in_d < input_depth && 
                                    in_h >= 0 && in_h < input_height && 
                                    in_w >= 0 && in_w < input_width) {
                                    
                                    for (int ic = 0; ic < in_channels; ic++) {
                                        int input_idx = b_idx * (in_channels * input_depth * input_height * input_width) +
                                                       ic * (input_depth * input_height * input_width) +
                                                       in_d * (input_height * input_width) +
                                                       in_h * input_width +
                                                       in_w;
                                                       
                                        int weight_idx = c_idx * (in_channels * kernel_d * kernel_h * kernel_w) +
                                                        ic * (kernel_d * kernel_h * kernel_w) +
                                                        kd * (kernel_h * kernel_w) +
                                                        kh * kernel_w +
                                                        kw;
                                                        
                                        conv_result += input[input_idx] * conv_weight[weight_idx];
                                    }
                                }
                            }
                        }
                    }
                    
                    // Apply division
                    conv_result /= divisor;
                    
                    if (conv_result > max_val) {
                        max_val = conv_result;
                    }
                }
            }
        }
    }
    
    // Adaptive average pooling approximation
    // For simplicity, we'll just use the max value scaled by a factor
    // In a production implementation, this would be a proper adaptive avg pool
    float avg_val = max_val; // Simplified for this example
    
    // Add bias
    avg_val += bias[c_idx];
    
    // Write result
    output[idx] = avg_val;
}

// Host function to launch kernel
void fused_op_forward(
    torch::Tensor input,
    torch::Tensor conv_weight,
    torch::Tensor conv_bias,
    float divisor,
    torch::Tensor bias,
    torch::Tensor output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_depth,
    int input_height,
    int input_width,
    int kernel_d,
    int kernel_h,
    int kernel_w,
    int conv_stride_d,
    int conv_stride_h,
    int conv_stride_w,
    int conv_pad_d,
    int conv_pad_h,
    int conv_pad_w,
    int pool_kernel_d,
    int pool_kernel_h,
    int pool_kernel_w,
    int pool_stride_d,
    int pool_stride_h,
    int pool_stride_w,
    int pool_pad_d,
    int pool_pad_h,
    int pool_pad_w,
    int global_avg_pool_d,
    int global_avg_pool_h,
    int global_avg_pool_w,
    int sum_dim
) {
    int conv_out_d = (input_depth + 2*conv_pad_d - kernel_d) / conv_stride_d + 1;
    int conv_out_h = (input_height + 2*conv_pad_h - kernel_h) / conv_stride_h + 1;
    int conv_out_w = (input_width + 2*conv_pad_w - kernel_w) / conv_stride_w + 1;
    
    int pool_out_d = (conv_out_d + 2*pool_pad_d - pool_kernel_d) / pool_stride_d + 1;
    int pool_out_h = (conv_out_h + 2*pool_pad_h - pool_kernel_h) / pool_stride_h + 1;
    int pool_out_w = (conv_out_w + 2*pool_pad_w - pool_kernel_w) / pool_stride_w + 1;
    
    int total_elements = batch_size * out_channels * pool_out_d * pool_out_h * pool_out_w;
    int blocks = (total_elements + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    fused_op_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        input.data_ptr<float>(),
        conv_weight.data_ptr<float>(),
        conv_bias.data_ptr<float>(),
        divisor,
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_depth,
        input_height,
        input_width,
        kernel_d, kernel_h, kernel_w,
        conv_stride_d, conv_stride_h, conv_stride_w,
        conv_pad_d, conv_pad_h, conv_pad_w,
        pool_kernel_d, pool_kernel_h, pool_kernel_w,
        pool_stride_d, pool_stride_h, pool_stride_w,
        pool_pad_d, pool_pad_h, pool_pad_w,
        global_avg_pool_d, global_avg_pool_h, global_avg_pool_w,
        sum_dim
    );
}
"""

# C++ binding code
cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(
    torch::Tensor input,
    torch::Tensor conv_weight,
    torch::Tensor conv_bias,
    float divisor,
    torch::Tensor bias,
    torch::Tensor output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_depth,
    int input_height,
    int input_width,
    int kernel_d,
    int kernel_h,
    int kernel_w,
    int conv_stride_d,
    int conv_stride_h,
    int conv_stride_w,
    int conv_pad_d,
    int conv_pad_h,
    int conv_pad_w,
    int pool_kernel_d,
    int pool_kernel_h,
    int pool_kernel_w,
    int pool_stride_d,
    int pool_stride_h,
    int pool_stride_w,
    int pool_pad_d,
    int pool_pad_h,
    int pool_pad_w,
    int global_avg_pool_d,
    int global_avg_pool_h,
    int global_avg_pool_w,
    int sum_dim
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused 3D Conv + Pool + Bias operation");
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
    # Move tensors to CUDA
    x = x.cuda()
    conv_weight = conv_weight.cuda()
    conv_bias = conv_bias.cuda()
    divisor = float(divisor)
    bias = bias.cuda()
    
    # Extract dimensions
    batch_size, in_channels, input_depth, input_height, input_width = x.shape
    out_channels = conv_weight.shape[0]
    kernel_d, kernel_h, kernel_w = conv_weight.shape[2], conv_weight.shape[3], conv_weight.shape[4]
    
    # Conv stride and padding
    conv_stride_d, conv_stride_h, conv_stride_w = conv_stride
    conv_pad_d, conv_pad_h, conv_pad_w = conv_padding
    
    # Pool parameters
    pool_kernel_d, pool_kernel_h, pool_kernel_w = max_pool_kernel_size
    pool_stride_d, pool_stride_h, pool_stride_w = max_pool_stride
    pool_pad_d, pool_pad_h, pool_pad_w = max_pool_padding
    
    # Global avg pool dimensions
    global_avg_pool_d, global_avg_pool_h, global_avg_pool_w = global_avg_pool_output_size
    
    # Calculate intermediate dimensions
    conv_out_d = (input_depth + 2*conv_pad_d - kernel_d) // conv_stride_d + 1
    conv_out_h = (input_height + 2*conv_pad_h - kernel_h) // conv_stride_h + 1
    conv_out_w = (input_width + 2*conv_pad_w - kernel_w) // conv_stride_w + 1
    
    pool_out_d = (conv_out_d + 2*pool_pad_d - pool_kernel_d) // pool_stride_d + 1
    pool_out_h = (conv_out_h + 2*pool_pad_h - pool_kernel_h) // pool_stride_h + 1
    pool_out_w = (conv_out_w + 2*pool_pad_w - pool_kernel_w) // pool_stride_w + 1
    
    # Create output tensor after pooling
    output = torch.empty(batch_size, out_channels, pool_out_d, pool_out_h, pool_out_w, device='cuda', dtype=torch.float32)
    
    # Call fused operation
    fused_ext.fused_op(
        x, conv_weight, conv_bias, divisor, bias, output,
        batch_size, in_channels, out_channels,
        input_depth, input_height, input_width,
        kernel_d, kernel_h, kernel_w,
        conv_stride_d, conv_stride_h, conv_stride_w,
        conv_pad_d, conv_pad_h, conv_pad_w,
        pool_kernel_d, pool_kernel_h, pool_kernel_w,
        pool_stride_d, pool_stride_h, pool_stride_w,
        pool_pad_d, pool_pad_h, pool_pad_w,
        global_avg_pool_d, global_avg_pool_h, global_avg_pool_w,
        sum_dim
    )
    
    # Perform adaptive average pooling to the specified size
    x = F.adaptive_avg_pool3d(output, global_avg_pool_output_size)
    
    # Sum reduction
    x = torch.sum(x, dim=sum_dim)
    
    return x

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
