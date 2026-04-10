# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_140117/code_12.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'output_padding', 'bias_shape']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'bias']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a 3D transposed convolution, followed by a sum, 
    a residual add, a multiplication, and another residual add.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))

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
    if 'bias' in flat_state:
        state_kwargs['bias'] = flat_state['bias']
    else:
        state_kwargs['bias'] = getattr(model, 'bias')
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

# Optimized CUDA kernels: custom conv_transpose3d + fused post-processing
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CUDA_1D_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

// Custom conv_transpose3d kernel for 3D convolution
__global__ void conv_transpose3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_depth,
    int input_height,
    int input_width,
    int output_depth,
    int output_height,
    int output_width,
    int kernel_size,
    int stride,
    int padding,
    int output_padding
) {
    int total_elements = batch_size * out_channels * output_depth * output_height * output_width;
    CUDA_1D_KERNEL_LOOP(idx, total_elements) {
        int w = idx % output_width;
        int h = (idx / output_width) % output_height;
        int d = (idx / (output_width * output_height)) % output_depth;
        int c = (idx / (output_width * output_height * output_depth)) % out_channels;
        int b = idx / (output_width * output_height * output_depth * out_channels);
        
        float sum = (bias != nullptr) ? bias[c] : 0.0f;
        
        // Compute input bounds for this output position
        int start_kd = max(0, (d + padding - kernel_size + 1 + stride - 1) / stride);
        int end_kd = min(kernel_size, (d + padding + stride) / stride);
        int start_kh = max(0, (h + padding - kernel_size + 1 + stride - 1) / stride);
        int end_kh = min(kernel_size, (h + padding + stride) / stride);
        int start_kw = max(0, (w + padding - kernel_size + 1 + stride - 1) / stride);
        int end_kw = min(kernel_size, (w + padding + stride) / stride);
        
        for (int kd = start_kd; kd < end_kd; ++kd) {
            for (int kh = start_kh; kh < end_kh; ++kh) {
                for (int kw = start_kw; kw < end_kw; ++kw) {
                    int input_d = (d + padding - kd) / stride;
                    int input_h = (h + padding - kh) / stride;
                    int input_w = (w + padding - kw) / stride;
                    
                    if (input_d >= 0 && input_d < input_depth &&
                        input_h >= 0 && input_h < input_height &&
                        input_w >= 0 && input_w < input_width) {
                        
                        int input_idx = b * (in_channels * input_depth * input_height * input_width) +
                                       ((0 * input_depth + input_d) * input_height + input_h) * input_width + input_w;
                        
                        int weight_idx = c * (in_channels * kernel_size * kernel_size * kernel_size) +
                                       (0 * kernel_size + kd) * kernel_size * kernel_size +
                                       kh * kernel_size + kw;
                        
                        for (int ic = 0; ic < in_channels; ++ic) {
                            sum += input[input_idx + ic * input_depth * input_height * input_width] *
                                   weight[weight_idx + ic * kernel_size * kernel_size * kernel_size];
                        }
                    }
                }
            }
        }
        
        output[idx] = sum;
    }
}

// Optimized fused post-processing kernel
__global__ void fused_post_conv_kernel(
    const float4* __restrict__ input,
    const float* __restrict__ bias,
    float4* __restrict__ output,
    int64_t num_elements_float4,
    int64_t spatial_size,
    int64_t out_channels
) {
    CUDA_1D_KERNEL_LOOP(idx, num_elements_float4) {
        // Calculate base channel index for this float4 element
        int64_t element_idx = idx * 4;
        int64_t base_channel_idx = (element_idx / spatial_size) % out_channels;
        
        float4 x_vec = input[idx];
        float b = bias[base_channel_idx];
        
        // Apply arithmetic: ((x + b) + x) * x + x = (2x + b) * x + x
        // Using FMA for better performance
        float4 result;
        result.x = fmaf(fmaf(2.0f, x_vec.x, b), x_vec.x, x_vec.x);
        result.y = fmaf(fmaf(2.0f, x_vec.y, b), x_vec.y, x_vec.y);
        result.z = fmaf(fmaf(2.0f, x_vec.z, b), x_vec.z, x_vec.z);
        result.w = fmaf(fmaf(2.0f, x_vec.w, b), x_vec.w, x_vec.w);
        
        output[idx] = result;
    }
}

void conv_transpose3d_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    int stride,
    int padding,
    int output_padding
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_depth = input.size(2);
    int input_height = input.size(3);
    int input_width = input.size(4);
    
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);
    
    int output_depth = (input_depth - 1) * stride - 2 * padding + kernel_size + output_padding;
    int output_height = (input_height - 1) * stride - 2 * padding + kernel_size + output_padding;
    int output_width = (input_width - 1) * stride - 2 * padding + kernel_size + output_padding;
    
    int threads_per_block = 256;
    int total_elements = batch_size * out_channels * output_depth * output_height * output_width;
    int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    conv_transpose3d_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_depth,
        input_height,
        input_width,
        output_depth,
        output_height,
        output_width,
        kernel_size,
        stride,
        padding,
        output_padding
    );
}

void fused_post_conv_forward(
    const torch::Tensor& input,
    const torch::Tensor& bias,
    torch::Tensor& output
) {
    int64_t num_elements = input.numel();
    int64_t num_elements_float4 = (num_elements + 3) / 4;
    int64_t spatial_size = input.size(2) * input.size(3) * input.size(4);
    int64_t out_channels = input.size(1);
    
    int threads_per_block = 256;
    int blocks = (num_elements_float4 + threads_per_block - 1) / threads_per_block;
    
    const float4* input_ptr = reinterpret_cast<const float4*>(input.data_ptr<float>());
    float4* output_ptr = reinterpret_cast<float4*>(output.data_ptr<float>());
    
    fused_post_conv_kernel<<<blocks, threads_per_block>>>(
        input_ptr,
        bias.data_ptr<float>(),
        output_ptr,
        num_elements_float4,
        spatial_size,
        out_channels
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void conv_transpose3d_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    int stride,
    int padding,
    int output_padding
);

void fused_post_conv_forward(const torch::Tensor& input, const torch::Tensor& bias, torch::Tensor& output);

torch::Tensor custom_conv_transpose3d(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    int stride,
    int padding,
    int output_padding
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_depth = input.size(2);
    int input_height = input.size(3);
    int input_width = input.size(4);
    
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);
    
    int output_depth = (input_depth - 1) * stride - 2 * padding + kernel_size + output_padding;
    int output_height = (input_height - 1) * stride - 2 * padding + kernel_size + output_padding;
    int output_width = (input_width - 1) * stride - 2 * padding + kernel_size + output_padding;
    
    auto output = torch::empty({batch_size, out_channels, output_depth, output_height, output_width}, 
                               torch::TensorOptions().dtype(input.dtype()).device(input.device()));
    
    conv_transpose3d_forward(input, weight, bias, output, stride, padding, output_padding);
    return output;
}

torch::Tensor fused_post_conv(const torch::Tensor& input, const torch::Tensor& bias) {
    auto output = torch::empty_like(input);
    fused_post_conv_forward(input, bias, output);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("custom_conv_transpose3d", &custom_conv_transpose3d, "Custom 3D convolution transpose");
    m.def("fused_post_conv", &fused_post_conv, "Fused post-conv arithmetic - direct global memory bias access");
}
"""

fused_ext = load_inline(
    name='fused_conv_ext',
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
    bias,
):
    # Perform the convolution using our custom kernel
    x = fused_ext.custom_conv_transpose3d(
        x, 
        conv_transpose_weight, 
        conv_transpose_bias,
        conv_transpose_stride[0],  # assuming uniform stride
        conv_transpose_padding[0],  # assuming uniform padding
        conv_transpose_output_padding[0]  # assuming uniform output padding
    )
    
    # Flatten bias for simplified kernel indexing
    bias_flat = bias.view(-1)
    
    # Use optimized fused kernel for the intensive post-processing element-wise ops
    return fused_ext.fused_post_conv(x, bias_flat)

batch_size = 16
in_channels = 32
out_channels = 64
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1, 1)

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape]

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width).cuda()]
