# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_133559/code_1.py
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

# Optimized CUDA kernels for both conv transpose and post-processing
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Conv transpose 3D kernel
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
    int out_depth,
    int out_height,
    int out_width,
    int kernel_size,
    int stride,
    int padding,
    int output_padding
) {
    // Calculate output indices
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output_elements = batch_size * out_channels * out_depth * out_height * out_width;
    
    if (out_idx >= total_output_elements) return;
    
    int w_out = out_idx % out_width;
    int h_out = (out_idx / out_width) % out_height;
    int d_out = (out_idx / (out_width * out_height)) % out_depth;
    int c_out = (out_idx / (out_width * out_height * out_depth)) % out_channels;
    int b = out_idx / (out_width * out_height * out_depth * out_channels);
    
    float sum = 0.0f;
    
    // Loop over kernel dimensions
    for (int k_d = 0; k_d < kernel_size; k_d++) {
        for (int k_h = 0; k_h < kernel_size; k_h++) {
            for (int k_w = 0; k_w < kernel_size; k_w++) {
                // Calculate corresponding input position
                int d_in = (d_out + padding - k_d);
                int h_in = (h_out + padding - k_h);
                int w_in = (w_out + padding - k_w);
                
                if (d_in % stride == 0 && h_in % stride == 0 && w_in % stride == 0) {
                    d_in /= stride;
                    h_in /= stride;
                    w_in /= stride;
                    
                    // Check bounds
                    if (d_in >= 0 && d_in < in_depth && 
                        h_in >= 0 && h_in < in_height && 
                        w_in >= 0 && w_in < in_width) {
                        
                        // Loop over input channels (assuming groups=1 for simplicity)
                        for (int c_in = 0; c_in < in_channels; c_in++) {
                            int input_idx = b * (in_channels * in_depth * in_height * in_width) +
                                          c_in * (in_depth * in_height * in_width) +
                                          d_in * (in_height * in_width) +
                                          h_in * in_width + w_in;
                                          
                            int weight_idx = c_out * (in_channels * kernel_size * kernel_size * kernel_size) +
                                           c_in * (kernel_size * kernel_size * kernel_size) +
                                           k_d * (kernel_size * kernel_size) +
                                           k_h * kernel_size + k_w;
                                           
                            sum += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
    }
    
    // Add bias
    sum += bias[c_out];
    
    output[out_idx] = sum;
}

// Optimized post-processing kernel using float4 vectorization
__global__ void fused_post_conv_kernel(
    const float4* __restrict__ input,
    const float* __restrict__ bias,
    float4* __restrict__ output,
    int64_t num_elements_float4,
    int64_t spatial_size,
    int64_t out_channels
) {
    // Grid-stride loop: each thread processes multiple elements
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; 
         idx < num_elements_float4; 
         idx += blockDim.x * gridDim.x) {
        
        // Calculate base channel index for the first element in the float4
        int64_t base_channel_idx = (idx * 4 / spatial_size) % out_channels;
        
        // Load 4 consecutive elements
        float4 x_vec = input[idx];
        float4 result;
        
        // Process each element with its corresponding bias: ((x + bias) + x) * x + x = (2x + bias) * x + x
        float bias_val = bias[base_channel_idx];
        result.x = (2.0f * x_vec.x + bias_val) * x_vec.x + x_vec.x;
        result.y = (2.0f * x_vec.y + bias_val) * x_vec.y + x_vec.y;
        result.z = (2.0f * x_vec.z + bias_val) * x_vec.z + x_vec.z;
        result.w = (2.0f * x_vec.w + bias_val) * x_vec.w + x_vec.w;
        
        // Store 4 consecutive results
        output[idx] = result;
    }
}

void fused_conv_transpose_post_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& conv_bias,
    const torch::Tensor& post_bias,
    torch::Tensor& output,
    int stride,
    int padding,
    int output_padding
) {
    auto input_sizes = input.sizes();
    auto weight_sizes = weight.sizes();
    
    int batch_size = input_sizes[0];
    int in_channels = input_sizes[1];
    int in_depth = input_sizes[2];
    int in_height = input_sizes[3];
    int in_width = input_sizes[4];
    
    int out_channels = weight_sizes[1];
    int kernel_size = weight_sizes[2];
    
    int out_depth = (in_depth - 1) * stride - 2 * padding + kernel_size + output_padding;
    int out_height = (in_height - 1) * stride - 2 * padding + kernel_size + output_padding;
    int out_width = (in_width - 1) * stride - 2 * padding + kernel_size + output_padding;
    
    // First, perform conv transpose
    int total_output_elements = batch_size * out_channels * out_depth * out_height * out_width;
    
    // Launch conv transpose kernel
    int threads_per_block = 256;
    int blocks = (total_output_elements + threads_per_block - 1) / threads_per_block;
    
    conv_transpose3d_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        conv_bias.data_ptr<float>(),
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
        kernel_size,
        stride,
        padding,
        output_padding
    );
    
    // Then apply post-processing with float4 vectorization
    int64_t num_elements = output.numel();
    int64_t num_elements_float4 = (num_elements + 3) / 4;  // Ceiling division
    int64_t spatial_size = out_depth * out_height * out_width;
    
    int post_blocks = min(68 * 4, (int)((num_elements_float4 + threads_per_block - 1) / threads_per_block));
    
    const float4* output_ptr = reinterpret_cast<const float4*>(output.data_ptr<float>());
    float4* output_ptr_out = reinterpret_cast<float4*>(output.data_ptr<float>());
    
    fused_post_conv_kernel<<<post_blocks, threads_per_block>>>(
        output_ptr,
        post_bias.data_ptr<float>(),
        output_ptr_out,
        num_elements_float4,
        spatial_size,
        out_channels
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_conv_transpose_post_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& conv_bias,
    const torch::Tensor& post_bias,
    torch::Tensor& output,
    int stride,
    int padding,
    int output_padding
);

torch::Tensor fused_conv_transpose_post(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& conv_bias,
    const torch::Tensor& post_bias,
    int stride,
    int padding,
    int output_padding
) {
    auto input_sizes = input.sizes();
    int batch_size = input_sizes[0];
    int in_channels = input_sizes[1];
    int in_depth = input_sizes[2];
    int in_height = input_sizes[3];
    int in_width = input_sizes[4];
    
    auto weight_sizes = weight.sizes();
    int out_channels = weight_sizes[1];
    int kernel_size = weight_sizes[2];
    
    int out_depth = (in_depth - 1) * stride - 2 * padding + kernel_size + output_padding;
    int out_height = (in_height - 1) * stride - 2 * padding + kernel_size + output_padding;
    int out_width = (in_width - 1) * stride - 2 * padding + kernel_size + output_padding;
    
    auto output = torch::empty({batch_size, out_channels, out_depth, out_height, out_width}, 
                              torch::TensorOptions().dtype(input.dtype()).device(input.device()));
    
    fused_conv_transpose_post_forward(input, weight, conv_bias, post_bias, output, stride, padding, output_padding);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_transpose_post", &fused_conv_transpose_post, "Fused conv transpose 3D and post-processing");
}
"""

fused_ext = load_inline(
    name='fused_conv_transpose_post_ext',
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
    # Flatten bias for simplified kernel indexing
    bias_flat = bias.view(-1)
    
    # Use optimized fused kernel for both conv transpose and post-processing
    return fused_ext.fused_conv_transpose_post(
        x, 
        conv_transpose_weight, 
        conv_transpose_bias, 
        bias_flat,
        conv_transpose_stride[0],  # Assuming uniform stride
        conv_transpose_padding[0],  # Assuming uniform padding
        conv_transpose_output_padding[0]  # Assuming uniform output padding
    )

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
