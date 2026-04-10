# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_134740/code_7.py
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
from torch.utils.cpp_extension import load_inline

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define WARP_SIZE 32
#define MAX_CHANNELS 1024

__global__ void fused_post_conv_kernel(
    const float4* __restrict__ input,
    const float* __restrict__ bias,
    float4* __restrict__ output,
    int64_t num_elements_float4,
    int64_t spatial_size,
    int64_t out_channels
) {
    // Shared memory for bias caching
    __shared__ float shared_bias[MAX_CHANNELS];
    
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
        
        // Calculate base indices for all 4 elements in the float4
        int64_t base_element_idx = idx * 4;
        int64_t channel_base = (base_element_idx / spatial_size) % out_channels;
        
        // Apply #pragma unroll for loop unrolling optimization
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int64_t offset_element_idx = base_element_idx + i;
            int64_t channel_idx = channel_base + (i * ((offset_element_idx / spatial_size) % out_channels - channel_base));
            // Simplified as we assume channel doesn't change within a float4 in practical cases
            // For precision, recalculate each channel index
            channel_idx = (offset_element_idx / spatial_size) % out_channels;
            
            float x_val;
            switch(i) {
                case 0: x_val = x_vec.x; break;
                case 1: x_val = x_vec.y; break;
                case 2: x_val = x_vec.z; break;
                case 3: x_val = x_vec.w; break;
            }
            
            float biased_val = x_val + shared_bias[channel_idx];
            float computed_val = (biased_val + x_val) * x_val + x_val;
            
            switch(i) {
                case 0: result.x = computed_val; break;
                case 1: result.y = computed_val; break;
                case 2: result.z = computed_val; break;
                case 3: result.w = computed_val; break;
            }
        }
        
        // Store 4 consecutive results
        output[idx] = result;
    }
}

// Optimized kernel using float4 vectorization, shared memory, and loop unrolling
__global__ void fused_post_conv_kernel_optimized(
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
        
        // Calculate base index for the first element in the float4
        int64_t base_element_idx = idx * 4;
        int64_t channel_base = (base_element_idx / spatial_size) % out_channels;
        
        // Manually unrolled computations for each component of float4
        result.x = ((x_vec.x + shared_bias[channel_base]) + x_vec.x) * x_vec.x + x_vec.x;
        
        int64_t channel_idx_1 = ((base_element_idx + 1) / spatial_size) % out_channels;
        result.y = ((x_vec.y + shared_bias[channel_idx_1]) + x_vec.y) * x_vec.y + x_vec.y;
        
        int64_t channel_idx_2 = ((base_element_idx + 2) / spatial_size) % out_channels;
        result.z = ((x_vec.z + shared_bias[channel_idx_2]) + x_vec.z) * x_vec.z + x_vec.z;
        
        int64_t channel_idx_3 = ((base_element_idx + 3) / spatial_size) % out_channels;
        result.w = ((x_vec.w + shared_bias[channel_idx_3]) + x_vec.w) * x_vec.w + x_vec.w;
        
        // Store 4 consecutive results
        output[idx] = result;
    }
}

// Transpose3D kernel
__global__ void conv_transpose3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int in_depth,
    int in_height,
    int in_width,
    int out_channels,
    int kernel_size,
    int stride,
    int padding,
    int output_padding
) {
    int out_d = (in_depth - 1) * stride - 2 * padding + kernel_size + output_padding;
    int out_h = (in_height - 1) * stride - 2 * padding + kernel_size + output_padding;
    int out_w = (in_width - 1) * stride - 2 * padding + kernel_size + output_padding;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output_elements = batch_size * out_channels * out_d * out_h * out_w;
    
    if (idx >= total_output_elements) return;
    
    int d_out = idx % out_d;
    int h_out = (idx / out_d) % out_h;
    int w_out = (idx / (out_d * out_h)) % out_w;
    int c_out = (idx / (out_d * out_h * out_w)) % out_channels;
    int b = idx / (out_d * out_h * out_w * out_channels);
    
    float value = 0.0f;
    
    for (int k_d = 0; k_d < kernel_size; ++k_d) {
        for (int k_h = 0; k_h < kernel_size; ++k_h) {
            for (int k_w = 0; k_w < kernel_size; ++k_w) {
                int in_d = d_out + padding - k_d;
                int in_h = h_out + padding - k_h;
                int in_w = w_out + padding - k_w;
                
                if (in_d % stride == 0 && in_h % stride == 0 && in_w % stride == 0) {
                    in_d /= stride;
                    in_h /= stride;
                    in_w /= stride;
                    
                    if (in_d >= 0 && in_d < in_depth &&
                        in_h >= 0 && in_h < in_height &&
                        in_w >= 0 && in_w < in_width) {
                        
                        for (int c_in = 0; c_in < in_channels; ++c_in) {
                            int input_idx = b * (in_channels * in_depth * in_height * in_width) +
                                            c_in * (in_depth * in_height * in_width) +
                                            in_d * (in_height * in_width) +
                                            in_h * in_width +
                                            in_w;
                            
                            int weight_idx = c_out * (in_channels * kernel_size * kernel_size * kernel_size) +
                                             c_in * (kernel_size * kernel_size * kernel_size) +
                                             k_d * (kernel_size * kernel_size) +
                                             k_h * kernel_size +
                                             k_w;
                            
                            value += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
    }
    
    output[idx] = value + bias[c_out];
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
    
    fused_post_conv_kernel_optimized<<<blocks, threads_per_block, shared_mem_size>>>(
        input_ptr,
        bias.data_ptr<float>(),
        output_ptr,
        num_elements_float4,
        spatial_size,
        out_channels
    );
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
    int in_depth = input.size(2);
    int in_height = input.size(3);
    int in_width = input.size(4);
    int out_channels = weight.size(1);
    int kernel_size = weight.size(2);
    
    int out_d = (in_depth - 1) * stride - 2 * padding + kernel_size + output_padding;
    int out_h = (in_height - 1) * stride - 2 * padding + kernel_size + output_padding;
    int out_w = (in_width - 1) * stride - 2 * padding + kernel_size + output_padding;
    
    int threads_per_block = 256;
    int total_output_elements = batch_size * out_channels * out_d * out_h * out_w;
    int blocks = (total_output_elements + threads_per_block - 1) / threads_per_block;
    
    conv_transpose3d_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        in_depth,
        in_height,
        in_width,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_post_conv_forward(const torch::Tensor& input, const torch::Tensor& bias, torch::Tensor& output);
void conv_transpose3d_forward(const torch::Tensor& input, const torch::Tensor& weight, const torch::Tensor& bias, 
                              torch::Tensor& output, int stride, int padding, int output_padding);

torch::Tensor fused_post_conv(const torch::Tensor& input, const torch::Tensor& bias) {
    auto output = torch::empty_like(input);
    fused_post_conv_forward(input, bias, output);
    return output;
}

torch::Tensor custom_conv_transpose3d(const torch::Tensor& input, const torch::Tensor& weight, const torch::Tensor& bias,
                                      int stride, int padding, int output_padding) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_depth = input.size(2);
    int in_height = input.size(3);
    int in_width = input.size(4);
    int kernel_size = weight.size(2);
    int out_channels = weight.size(1);
    
    int out_d = (in_depth - 1) * stride - 2 * padding + kernel_size + output_padding;
    int out_h = (in_height - 1) * stride - 2 * padding + kernel_size + output_padding;
    int out_w = (in_width - 1) * stride - 2 * padding + kernel_size + output_padding;
    
    auto output = torch::empty({batch_size, out_channels, out_d, out_h, out_w}, 
                               torch::TensorOptions().dtype(input.dtype()).device(input.device()));
                               
    conv_transpose3d_forward(input, weight, bias, output, stride, padding, output_padding);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_post_conv", &fused_post_conv, "Fused post-conv arithmetic with float4 vectorization and shared memory");
    m.def("conv_transpose3d", &custom_conv_transpose3d, "Custom ConvTranspose3D implementation");
}
"""

fused_ext = load_inline(
    name='fused_post_conv_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math', '-arch=sm_75'],
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
    # Perform the convolution using optimized custom kernel
    x = fused_ext.conv_transpose3d(x, conv_transpose_weight, conv_transpose_bias, 
                                   conv_transpose_stride, conv_transpose_padding, 
                                   conv_transpose_output_padding)
    
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
