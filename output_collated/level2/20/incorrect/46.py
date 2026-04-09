# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_133559/code_2.py
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

# Optimized CUDA kernel with improved memory coalescing
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
    // Shared memory for bias caching
    extern __shared__ float shared_bias[];
    
    // Cooperative loading of bias into shared memory
    int tid = threadIdx.x;
    for (int i = tid; i < out_channels; i += blockDim.x) {
        shared_bias[i] = bias[i];
    }
    __syncthreads();
    
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements_float4) {
        // Load 4 consecutive elements
        float4 x_vec = input[idx];
        float4 result;
        
        // Calculate channel index for this float4 (all 4 elements have same channel)
        int64_t base_element_idx = idx * 4;
        int64_t channel_idx = (base_element_idx / spatial_size) % out_channels;
        float channel_bias = shared_bias[channel_idx];
        
        // Process all 4 elements with the same bias (improves coalescing)
        result.x = ((x_vec.x + channel_bias) + x_vec.x) * x_vec.x + x_vec.x;
        result.y = ((x_vec.y + channel_bias) + x_vec.y) * x_vec.y + x_vec.y;
        result.z = ((x_vec.z + channel_bias) + x_vec.z) * x_vec.z + x_vec.z;
        result.w = ((x_vec.w + channel_bias) + x_vec.w) * x_vec.w + x_vec.w;
        
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
    int64_t num_elements_float4 = num_elements / 4;  // Exact division since we ensure alignment
    
    int64_t spatial_size = input.size(2) * input.size(3) * input.size(4);
    int64_t out_channels = input.size(1);
    
    int threads_per_block = 256;
    int blocks = (num_elements_float4 + threads_per_block - 1) / threads_per_block;
    
    // Cast pointers to float4 for vectorized access
    const float4* input_ptr = reinterpret_cast<const float4*>(input.data_ptr<float>());
    float4* output_ptr = reinterpret_cast<float4*>(output.data_ptr<float>());
    
    // Shared memory size for bias
    size_t shared_mem_size = out_channels * sizeof(float);
    
    fused_post_conv_kernel<<<blocks, threads_per_block, shared_mem_size>>>(
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
    m.def("fused_post_conv", &fused_post_conv, "Fused post-conv arithmetic with improved memory coalescing");
}
"""

fused_ext = load_inline(
    name='fused_post_conv_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# Custom CUDA kernel for conv transpose 3D
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
    int in_depth,
    int in_height,
    int in_width,
    int kernel_size,
    int stride,
    int padding,
    int output_padding
) {
    int out_depth = (in_depth - 1) * stride - 2 * padding + kernel_size + output_padding;
    int out_height = (in_height - 1) * stride - 2 * padding + kernel_size + output_padding;
    int out_width = (in_width - 1) * stride - 2 * padding + kernel_size + output_padding;
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * out_depth * out_height * out_width;
    
    if (tid < total_elements) {
        int w = tid % out_width;
        int h = (tid / out_width) % out_height;
        int d = (tid / (out_width * out_height)) % out_depth;
        int oc = (tid / (out_width * out_height * out_depth)) % out_channels;
        int b = tid / (out_width * out_height * out_depth * out_channels);
        
        float sum = 0.0f;
        
        // Loop over input and kernel
        for (int ic = 0; ic < in_channels; ic++) {
            for (int kd = 0; kd < kernel_size; kd++) {
                for (int kh = 0; kh < kernel_size; kh++) {
                    for (int kw = 0; kw < kernel_size; kw++) {
                        int in_d = d + padding - kd;
                        int in_h = h + padding - kh;
                        int in_w = w + padding - kw;
                        
                        if (in_d % stride == 0 && in_h % stride == 0 && in_w % stride == 0) {
                            in_d /= stride;
                            in_h /= stride;
                            in_w /= stride;
                            
                            if (in_d >= 0 && in_d < in_depth && 
                                in_h >= 0 && in_h < in_height && 
                                in_w >= 0 && in_w < in_width) {
                                
                                int input_idx = b * (in_channels * in_depth * in_height * in_width) +
                                               ic * (in_depth * in_height * in_width) +
                                               in_d * (in_height * in_width) +
                                               in_h * in_width + in_w;
                               
                                int weight_idx = oc * (in_channels * kernel_size * kernel_size * kernel_size) +
                                                ic * (kernel_size * kernel_size * kernel_size) +
                                                kd * (kernel_size * kernel_size) +
                                                kh * kernel_size + kw;
                                
                                sum += input[input_idx] * weight[weight_idx];
                            }
                        }
                    }
                }
            }
        }
        
        output[tid] = sum + bias[oc];
    }
}

torch::Tensor conv_transpose3d_cuda(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    int stride,
    int padding,
    int output_padding
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_depth = input.size(2);
    int in_height = input.size(3);
    int in_width = input.size(4);
    int kernel_size = weight.size(2);
    int out_channels = weight.size(0);
    
    int out_depth = (in_depth - 1) * stride - 2 * padding + kernel_size + output_padding;
    int out_height = (in_height - 1) * stride - 2 * padding + kernel_size + output_padding;
    int out_width = (in_width - 1) * stride - 2 * padding + kernel_size + output_padding;
    
    auto output = torch::empty({batch_size, out_channels, out_depth, out_height, out_width}, 
                              torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    
    int total_elements = batch_size * out_channels * out_depth * out_height * out_width;
    int threads_per_block = 256;
    int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    conv_transpose3d_kernel<<<blocks, threads_per_block>>>(
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
        kernel_size,
        stride,
        padding,
        output_padding
    );
    
    return output;
}
"""

conv_transpose3d_cpp = r"""
#include <torch/extension.h>

torch::Tensor conv_transpose3d_cuda(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    int stride,
    int padding,
    int output_padding
);

torch::Tensor conv_transpose3d(const torch::Tensor& input,
                               const torch::Tensor& weight,
                               const torch::Tensor& bias,
                               int stride,
                               int padding,
                               int output_padding) {
    return conv_transpose3d_cuda(input, weight, bias, stride, padding, output_padding);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_transpose3d", &conv_transpose3d, "3D Convolution Transpose CUDA implementation");
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
    # Perform the convolution using our custom CUDA kernel
    x = conv_transpose3d_ext.conv_transpose3d(
        x, conv_transpose_weight, conv_transpose_bias,
        conv_transpose_stride[0], conv_transpose_padding[0], conv_transpose_output_padding[0]
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
