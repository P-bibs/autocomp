# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_140117/code_11.py
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

# Optimized CUDA kernel using register caching, loop unrolling and efficient memory access
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__global__ void fused_post_conv_kernel(
    const float4* __restrict__ input,
    const float* __restrict__ bias,
    float4* __restrict__ output,
    int64_t num_elements_float4,
    int64_t spatial_size,
    int64_t out_channels
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_elements_float4) {
        // Load 4 consecutive elements
        float4 x_vec = input[idx];
        float4 result;
        
        // Calculate base indices for all 4 elements in the float4
        int64_t base_element_idx = idx * 4;
        int64_t channel_idx_0 = (base_element_idx / spatial_size) % out_channels;
        int64_t channel_idx_1 = ((base_element_idx + 1) / spatial_size) % out_channels;
        int64_t channel_idx_2 = ((base_element_idx + 2) / spatial_size) % out_channels;
        int64_t channel_idx_3 = ((base_element_idx + 3) / spatial_size) % out_channels;
        
        // Load bias values into registers (register caching)
        float bias_0 = bias[channel_idx_0];
        float bias_1 = bias[channel_idx_1];
        float bias_2 = bias[channel_idx_2];
        float bias_3 = bias[channel_idx_3];
        
        // Process each element with register-cached bias
        // Simplified operation: fused_op(x) = x * (x + bias)
        result.x = x_vec.x * (x_vec.x + bias_0);
        result.y = x_vec.y * (x_vec.y + bias_1);
        result.z = x_vec.z * (x_vec.z + bias_2);
        result.w = x_vec.w * (x_vec.w + bias_3);
        
        // Store 4 consecutive results
        output[idx] = result;
    }
}

void fused_post_conv_forward(
    const torch::Tensor& input,
    const torch::Tensor& bias,
    torch::Tensor& output
) {
    // Calculate number of float4 elements (total elements / 4)
    int64_t num_elements = input.numel();
    int64_t num_elements_float4 = num_elements / 4;
    
    int64_t spatial_size = input.size(2) * input.size(3) * input.size(4);
    int64_t out_channels = input.size(1);
    
    int threads_per_block = 256;
    int blocks = (num_elements_float4 + threads_per_block - 1) / threads_per_block;
    
    // Cast pointers to float4 for vectorized access
    const float4* input_ptr = reinterpret_cast<const float4*>(input.data_ptr<float>());
    float4* output_ptr = reinterpret_cast<float4*>(output.data_ptr<float>());
    
    // No shared memory needed - using register caching instead
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

void fused_post_conv_forward(const torch::Tensor& input, const torch::Tensor& bias, torch::Tensor& output);

torch::Tensor fused_post_conv(const torch::Tensor& input, const torch::Tensor& bias) {
    auto output = torch::empty_like(input);
    fused_post_conv_forward(input, bias, output);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_post_conv", &fused_post_conv, "Fused post-conv with register caching and optimized loop unrolling");
}
"""

fused_ext = load_inline(
    name='fused_post_conv_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# Custom ConvTranspose3d CUDA kernel
conv_transpose3d_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

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
    int oc = blockIdx.x;
    int od = blockIdx.y;
    int oh = blockIdx.z;
    int ow_thread = threadIdx.x;
    
    if (oc >= out_channels || od >= output_depth || oh >= output_height) return;
    
    int elements_per_thread = (output_width + blockDim.x - 1) / blockDim.x;
    for (int k = 0; k < elements_per_thread; k++) {
        int ow = ow_thread + k * blockDim.x;
        if (ow >= output_width) continue;
        
        float sum = 0.0f;
        
        // Compute input ranges
        int d_start = max(0, (od + padding - kernel_size + 1 + stride - 1) / stride);
        int d_end = min(input_depth, (od + padding) / stride + 1);
        int h_start = max(0, (oh + padding - kernel_size + 1 + stride - 1) / stride);
        int h_end = min(input_height, (oh + padding) / stride + 1);
        int w_start = max(0, (ow + padding - kernel_size + 1 + stride - 1) / stride);
        int w_end = min(input_width, (ow + padding) / stride + 1);
        
        for (int ic = 0; ic < in_channels; ic++) {
            for (int kd = max(0, od + padding - (d_end - 1) * stride); kd < kernel_size; kd++) {
                for (int kh = max(0, oh + padding - (h_end - 1) * stride); kh < kernel_size; kh++) {
                    for (int kw = max(0, ow + padding - (w_end - 1) * stride); kw < kernel_size; kw++) {
                        for (int id = d_start; id < d_end; id++) {
                            for (int ih = h_start; ih < h_end; ih++) {
                                for (int iw = w_start; iw < w_end; iw++) {
                                    if (id * stride + kd - padding == od &&
                                        ih * stride + kh - padding == oh &&
                                        iw * stride + kw - padding == ow) {
                                        int input_idx = ((ic * input_depth + id) * input_height + ih) * input_width + iw;
                                        int weight_idx = (((oc * in_channels + ic) * kernel_size + kd) * kernel_size + kh) * kernel_size + kw;
                                        sum += input[input_idx] * weight[weight_idx];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        int output_idx = (((oc * output_depth + od) * output_height + oh) * output_width + ow);
        output[output_idx] = sum + bias[oc];
    }
}

void conv_transpose3d_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    int64_t stride,
    int64_t padding,
    int64_t output_padding
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_depth = input.size(2);
    int input_height = input.size(3);
    int input_width = input.size(4);
    
    int out_channels = weight.size(1);
    int kernel_size = weight.size(2);
    
    int output_depth = (input_depth - 1) * stride - 2 * padding + kernel_size + output_padding;
    int output_height = (input_height - 1) * stride - 2 * padding + kernel_size + output_padding;
    int output_width = (input_width - 1) * stride - 2 * padding + kernel_size + output_padding;
    
    dim3 grid(out_channels, output_depth, output_height);
    dim3 block(min(1024, output_width));
    
    conv_transpose3d_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
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
"""

conv_transpose3d_cpp = r"""
#include <torch/extension.h>

void conv_transpose3d_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    int64_t stride,
    int64_t padding,
    int64_t output_padding
);

torch::Tensor conv_transpose3d_custom(const torch::Tensor& input, 
                                      const torch::Tensor& weight, 
                                      const torch::Tensor& bias,
                                      int64_t stride,
                                      int64_t padding,
                                      int64_t output_padding) {
    int input_depth = input.size(2);
    int input_height = input.size(3);
    int input_width = input.size(4);
    int kernel_size = weight.size(2);
    int output_depth = (input_depth - 1) * stride - 2 * padding + kernel_size + output_padding;
    int output_height = (input_height - 1) * stride - 2 * padding + kernel_size + output_padding;
    int output_width = (input_width - 1) * stride - 2 * padding + kernel_size + output_padding;
    
    auto output = torch::empty({input.size(0), weight.size(1), output_depth, output_height, output_width}, 
                               torch::dtype(input.dtype()).device(input.device()));
    
    conv_transpose3d_forward(input, weight, bias, output, stride, padding, output_padding);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_transpose3d_custom", &conv_transpose3d_custom, "Custom ConvTranspose3d implementation");
}
"""

conv_transpose3d_ext = load_inline(
    name='conv_transpose3d_ext',
    cpp_sources=conv_transpose3d_cpp,
    cuda_sources=conv_transpose3d_kernel,
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
    # Perform the convolution using custom CUDA kernel
    x = conv_transpose3d_ext.conv_transpose3d_custom(
        x, conv_transpose_weight, conv_transpose_bias,
        conv_transpose_stride[0],  # Assuming uniform stride
        conv_transpose_padding[0],  # Assuming uniform padding
        conv_transpose_output_padding[0]  # Assuming uniform output padding
    )
    
    # Flatten bias for simplified kernel indexing
    bias_flat = bias.view(-1)
    
    # Ensure input is contiguous and properly aligned for float4 access
    x = x.contiguous()
    
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
