# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_030214/code_0.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'max_pool1_kernel_size', 'max_pool1_stride', 'max_pool1_padding', 'max_pool1_dilation', 'max_pool1_ceil_mode', 'max_pool1_return_indices', 'max_pool2_kernel_size', 'max_pool2_stride', 'max_pool2_padding', 'max_pool2_dilation', 'max_pool2_ceil_mode', 'max_pool2_return_indices']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a 3D transposed convolution, followed by two max pooling layers and a sum operation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.max_pool1 = nn.MaxPool3d(kernel_size=2)
        self.max_pool2 = nn.MaxPool3d(kernel_size=3)

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
    # State for conv_transpose (nn.ConvTranspose3d)
    if 'conv_transpose_weight' in flat_state:
        state_kwargs['conv_transpose_weight'] = flat_state['conv_transpose_weight']
    else:
        state_kwargs['conv_transpose_weight'] = getattr(model.conv_transpose, 'weight', None)
    if 'conv_transpose_bias' in flat_state:
        state_kwargs['conv_transpose_bias'] = flat_state['conv_transpose_bias']
    else:
        state_kwargs['conv_transpose_bias'] = getattr(model.conv_transpose, 'bias', None)
    state_kwargs['conv_transpose_stride'] = model.conv_transpose.stride
    state_kwargs['conv_transpose_padding'] = model.conv_transpose.padding
    state_kwargs['conv_transpose_output_padding'] = model.conv_transpose.output_padding
    state_kwargs['conv_transpose_groups'] = model.conv_transpose.groups
    state_kwargs['conv_transpose_dilation'] = model.conv_transpose.dilation
    # State for max_pool1 (nn.MaxPool3d)
    state_kwargs['max_pool1_kernel_size'] = model.max_pool1.kernel_size
    state_kwargs['max_pool1_stride'] = model.max_pool1.stride
    state_kwargs['max_pool1_padding'] = model.max_pool1.padding
    state_kwargs['max_pool1_dilation'] = model.max_pool1.dilation
    state_kwargs['max_pool1_ceil_mode'] = model.max_pool1.ceil_mode
    state_kwargs['max_pool1_return_indices'] = model.max_pool1.return_indices
    # State for max_pool2 (nn.MaxPool3d)
    state_kwargs['max_pool2_kernel_size'] = model.max_pool2.kernel_size
    state_kwargs['max_pool2_stride'] = model.max_pool2.stride
    state_kwargs['max_pool2_padding'] = model.max_pool2.padding
    state_kwargs['max_pool2_dilation'] = model.max_pool2.dilation
    state_kwargs['max_pool2_ceil_mode'] = model.max_pool2.ceil_mode
    state_kwargs['max_pool2_return_indices'] = model.max_pool2.return_indices
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

# Fused ConvTranspose3d + MaxPool3d CUDA kernel
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAMathCompat.h>

#define CUDA_1D_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

__global__ void fused_conv_transpose_pool3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_depth,
    int in_height,
    int in_width,
    int out_depth,
    int out_height,
    int out_width,
    int kernel_d,
    int kernel_h,
    int kernel_w,
    int stride_d,
    int stride_h,
    int stride_w,
    int padding_d,
    int padding_h,
    int padding_w,
    int pool_kernel_size,
    int pool_stride,
    int pool_padding
) {
    // Calculate output indices
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output_elements = batch_size * out_channels * 
                               ((out_depth + pool_stride - 1) / pool_stride) * 
                               ((out_height + pool_stride - 1) / pool_stride) * 
                               ((out_width + pool_stride - 1) / pool_stride);
    
    if (out_idx >= total_output_elements) return;
    
    // Decompose output index
    int temp = out_idx;
    int out_w_idx = temp % ((out_width + pool_stride - 1) / pool_stride);
    temp /= ((out_width + pool_stride - 1) / pool_stride);
    int out_h_idx = temp % ((out_height + pool_stride - 1) / pool_stride);
    temp /= ((out_height + pool_stride - 1) / pool_stride);
    int out_d_idx = temp % ((out_depth + pool_stride - 1) / pool_stride);
    temp /= ((out_depth + pool_stride - 1) / pool_stride);
    int out_c_idx = temp % out_channels;
    int batch_idx = temp / out_channels;
    
    if (batch_idx >= batch_size) return;
    
    // Calculate the actual pooling region start coordinates
    int pool_out_d = out_d_idx * pool_stride;
    int pool_out_h = out_h_idx * pool_stride;
    int pool_out_w = out_w_idx * pool_stride;
    
    // Perform convolution + pooling in one step
    float max_val = -1e30f;
    bool found_valid = false;
    
    // Iterate through the pooling window
    for (int pd = 0; pd < pool_kernel_size && (pool_out_d + pd) < out_depth; pd++) {
        for (int ph = 0; ph < pool_kernel_size && (pool_out_h + ph) < out_height; ph++) {
            for (int pw = 0; pw < pool_kernel_size && (pool_out_w + pw) < out_width; pw++) {
                int conv_d = pool_out_d + pd;
                int conv_h = pool_out_h + ph;
                int conv_w = pool_out_w + pw;
                
                // Perform transposed convolution calculation for this point
                float conv_sum = 0.0f;
                
                // Iterate through kernel
                for (int kd = 0; kd < kernel_d; kd++) {
                    for (int kh = 0; kh < kernel_h; kh++) {
                        for (int kw = 0; kw < kernel_w; kw++) {
                            // Calculate input position
                            int in_d = conv_d + padding_d - kd;
                            int in_h = conv_h + padding_h - kh;
                            int in_w = conv_w + padding_w - kw;
                            
                            // Check bounds with stride adjustment
                            if (in_d >= 0 && in_d < in_depth * stride_d && in_d % stride_d == 0 &&
                                in_h >= 0 && in_h < in_height * stride_h && in_h % stride_h == 0 &&
                                in_w >= 0 && in_w < in_width * stride_w && in_w % stride_w == 0) {
                                
                                int act_in_d = in_d / stride_d;
                                int act_in_h = in_h / stride_h;
                                int act_in_w = in_w / stride_w;
                                
                                if (act_in_d < in_depth && act_in_h < in_height && act_in_w < in_width) {
                                    // Loop through input channels
                                    for (int ic = 0; ic < in_channels; ic++) {
                                        int input_idx = batch_idx * in_channels * in_depth * in_height * in_width +
                                                       ic * in_depth * in_height * in_width +
                                                       act_in_d * in_height * in_width +
                                                       act_in_h * in_width +
                                                       act_in_w;
                                        
                                        int weight_idx = out_c_idx * in_channels * kernel_d * kernel_h * kernel_w +
                                                        ic * kernel_d * kernel_h * kernel_w +
                                                        kd * kernel_h * kernel_w +
                                                        kh * kernel_w +
                                                        kw;
                                        
                                        conv_sum += input[input_idx] * weight[weight_idx];
                                    }
                                }
                            }
                        }
                    }
                }
                
                // Add bias
                conv_sum += bias[out_c_idx];
                
                // Update max
                if (!found_valid || conv_sum > max_val) {
                    max_val = conv_sum;
                    found_valid = true;
                }
            }
        }
    }
    
    if (found_valid) {
        output[out_idx] = max_val;
    }
}

void fused_conv_transpose_pool3d_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    int kernel_d,
    int kernel_h,
    int kernel_w,
    int stride_d,
    int stride_h,
    int stride_w,
    int padding_d,
    int padding_h,
    int padding_w,
    int pool_kernel_size,
    int pool_stride,
    int pool_padding
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_depth = input.size(2);
    int in_height = input.size(3);
    int in_width = input.size(4);
    int out_channels = weight.size(0);
    
    int out_depth = (in_depth - 1) * stride_d - 2 * padding_d + kernel_d;
    int out_height = (in_height - 1) * stride_h - 2 * padding_h + kernel_h;
    int out_width = (in_width - 1) * stride_w - 2 * padding_w + kernel_w;
    
    int pooled_depth = (out_depth + 2 * pool_padding - pool_kernel_size) / pool_stride + 1;
    int pooled_height = (out_height + 2 * pool_padding - pool_kernel_size) / pool_stride + 1;
    int pooled_width = (out_width + 2 * pool_padding - pool_kernel_size) / pool_stride + 1;
    
    int total_output_elements = batch_size * out_channels * pooled_depth * pooled_height * pooled_width;
    
    const int threads = 512;
    const int blocks = (total_output_elements + threads - 1) / threads;
    
    fused_conv_transpose_pool3d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        in_depth,
        in_height,
        in_width,
        out_depth,
        out_height,
        out_width,
        kernel_d,
        kernel_h,
        kernel_w,
        stride_d,
        stride_h,
        stride_w,
        padding_d,
        padding_h,
        padding_w,
        pool_kernel_size,
        pool_stride,
        pool_padding
    );
    
    cudaDeviceSynchronize();
}
"""

# C++ interface
cpp_source = r"""
#include <torch/extension.h>

void fused_conv_transpose_pool3d_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    int kernel_d,
    int kernel_h,
    int kernel_w,
    int stride_d,
    int stride_h,
    int stride_w,
    int padding_d,
    int padding_h,
    int padding_w,
    int pool_kernel_size,
    int pool_stride,
    int pool_padding
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_transpose_pool3d", &fused_conv_transpose_pool3d_forward, "Fused ConvTranspose3d + MaxPool3d");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_conv_transpose_pool3d_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(
    x,
    *,
    conv_transpose_weight,
    conv_transpose_bias,
    conv_transpose_stride,
    conv_transpose_padding,
    conv_transpose_output_padding,
    conv_transpose_groups,
    conv_transpose_dilation,
    max_pool1_kernel_size,
    max_pool1_stride,
    max_pool1_padding,
    max_pool1_dilation,
    max_pool1_ceil_mode,
    max_pool1_return_indices,
    max_pool2_kernel_size,
    max_pool2_stride,
    max_pool2_padding,
    max_pool2_dilation,
    max_pool2_ceil_mode,
    max_pool2_return_indices,
):
    # Use fused operation for first conv transpose + max pool
    batch_size = x.shape[0]
    in_channels = x.shape[1]
    in_depth, in_height, in_width = x.shape[2], x.shape[3], x.shape[4]
    
    # Handle non-uniform stride, padding, and kernel size
    kernel_d, kernel_h, kernel_w = conv_transpose_weight.shape[2], conv_transpose_weight.shape[3], conv_transpose_weight.shape[4]
    stride_d, stride_h, stride_w = conv_transpose_stride[0], conv_transpose_stride[1], conv_transpose_stride[2]
    padding_d, padding_h, padding_w = conv_transpose_padding[0], conv_transpose_padding[1], conv_transpose_padding[2]
    
    out_depth = (in_depth - 1) * stride_d - 2 * padding_d + kernel_d
    out_height = (in_height - 1) * stride_h - 2 * padding_h + kernel_h
    out_width = (in_width - 1) * stride_w - 2 * padding_w + kernel_w
    
    pooled_depth = (out_depth + 2 * max_pool1_padding - max_pool1_kernel_size) // max_pool1_stride + 1
    pooled_height = (out_height + 2 * max_pool1_padding - max_pool1_kernel_size) // max_pool1_stride + 1
    pooled_width = (out_width + 2 * max_pool1_padding - max_pool1_kernel_size) // max_pool1_stride + 1
    
    # Create output tensor for fused operation
    fused_output = torch.empty(
        batch_size, 
        conv_transpose_weight.shape[0], 
        pooled_depth, 
        pooled_height, 
        pooled_width,
        device=x.device, 
        dtype=x.dtype
    )
    
    # Call fused kernel
    fused_ext.fused_conv_transpose_pool3d(
        x.contiguous(),
        conv_transpose_weight.contiguous(),
        conv_transpose_bias.contiguous(),
        fused_output,
        kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w,
        max_pool1_kernel_size,
        max_pool1_stride,
        max_pool1_padding
    )
    
    # Apply second max pooling
    x = F.max_pool3d(fused_output, kernel_size=max_pool2_kernel_size, stride=max_pool2_stride, 
                     padding=max_pool2_padding, dilation=max_pool2_dilation, 
                     ceil_mode=max_pool2_ceil_mode, return_indices=max_pool2_return_indices)
    
    # Apply sum reduction
    x = torch.sum(x, dim=1, keepdim=True)
    return x

batch_size = 16
in_channels = 32
out_channels = 64
depth, height, width = 32, 32, 32
kernel_size = 5
stride = 2
padding = 2

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]
