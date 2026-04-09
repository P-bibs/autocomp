# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_133559/code_13.py
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

# Optimized CUDA kernel using grid-stride loops and float4 vectorization
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
    // Grid-stride loop: each thread processes multiple elements
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; 
         idx < num_elements_float4; 
         idx += blockDim.x * gridDim.x) {
        
        // Calculate base channel index for the first element in the float4
        int64_t base_channel_idx = (idx * 4 / spatial_size) % out_channels;
        
        // Load 4 consecutive elements
        float4 x_vec = input[idx];
        float4 result;
        
        // Process each element with its corresponding bias
        result.x = ((x_vec.x + bias[base_channel_idx]) + x_vec.x) * x_vec.x + x_vec.x;
        result.y = ((x_vec.y + bias[base_channel_idx]) + x_vec.y) * x_vec.y + x_vec.y;
        result.z = ((x_vec.z + bias[base_channel_idx]) + x_vec.z) * x_vec.z + x_vec.z;
        result.w = ((x_vec.w + bias[base_channel_idx]) + x_vec.w) * x_vec.w + x_vec.w;
        
        // Store 4 consecutive results
        output[idx] = result;
    }
}

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
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_out_elements = batch_size * out_channels * out_depth * out_height * out_width;
    
    if (out_idx >= total_out_elements) return;
    
    int w_out = out_idx % out_width;
    out_idx /= out_width;
    int h_out = out_idx % out_height;
    out_idx /= out_height;
    int d_out = out_idx % out_depth;
    out_idx /= out_depth;
    int c_out = out_idx % out_channels;
    int b = out_idx / out_channels;
    
    float sum = 0.0f;
    
    // Compute input coordinates that would contribute to this output
    for (int kd = 0; kd < kernel_size; kd++) {
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                int d_in = d_out + padding - kd;
                int h_in = h_out + padding - kh;
                int w_in = w_out + padding - kw;
                
                // Check if this is a valid input position
                if (d_in % stride == 0 && h_in % stride == 0 && w_in % stride == 0) {
                    d_in /= stride;
                    h_in /= stride;
                    w_in /= stride;
                    
                    if (d_in >= 0 && d_in < in_depth &&
                        h_in >= 0 && h_in < in_height &&
                        w_in >= 0 && w_in < in_width) {
                        
                        for (int c_in = 0; c_in < in_channels; c_in++) {
                            int input_idx = b * (in_channels * in_depth * in_height * in_width) +
                                            c_in * (in_depth * in_height * in_width) +
                                            d_in * (in_height * in_width) +
                                            h_in * in_width +
                                            w_in;
                            
                            int weight_idx = c_out * (in_channels * kernel_size * kernel_size * kernel_size) +
                                             c_in * (kernel_size * kernel_size * kernel_size) +
                                             kd * (kernel_size * kernel_size) +
                                             kh * kernel_size +
                                             kw;
                            
                            sum += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
    }
    
    output[out_idx * (out_channels * out_depth * out_height * out_width) +
           c_out * (out_depth * out_height * out_width) +
           d_out * (out_height * out_width) +
           h_out * out_width +
           w_out] = sum + bias[c_out];
}

void fused_post_conv_forward(
    const torch::Tensor& input,
    const torch::Tensor& bias,
    torch::Tensor& output
) {
    // Calculate number of float4 elements (total elements / 4)
    int64_t num_elements = input.numel();
    int64_t num_elements_float4 = (num_elements + 3) / 4;  // Ceiling division
    
    int64_t spatial_size = input.size(2) * input.size(3) * input.size(4);
    int64_t out_channels = input.size(1);
    
    // Use grid-stride loop approach with optimal grid size
    int threads_per_block = 256;
    int blocks_per_sm = 4;  // Heuristic for good occupancy
    int num_sms = 68;       // RTX 2080 Ti has 68 SMs
    int blocks = min(num_sms * blocks_per_sm, 
                     (int)((num_elements_float4 + threads_per_block - 1) / threads_per_block));
    
    // Cast pointers to float4 for vectorized access
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
    
    int out_depth = (in_depth - 1) * stride - 2 * padding + kernel_size + output_padding;
    int out_height = (in_height - 1) * stride - 2 * padding + kernel_size + output_padding;
    int out_width = (in_width - 1) * stride - 2 * padding + kernel_size + output_padding;
    
    int total_out_elements = batch_size * out_channels * out_depth * out_height * out_width;
    
    int threads_per_block = 256;
    int blocks = (total_out_elements + threads_per_block - 1) / threads_per_block;
    
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
        out_depth,
        out_height,
        out_width,
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

void conv_transpose3d(const torch::Tensor& input, const torch::Tensor& weight, const torch::Tensor& bias,
                      torch::Tensor& output, int stride, int padding, int output_padding) {
    conv_transpose3d_forward(input, weight, bias, output, stride, padding, output_padding);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_post_conv", &fused_post_conv, "Fused post-conv arithmetic with float4 vectorization");
    m.def("conv_transpose3d", &conv_transpose3d, "3D transposed convolution");
}
"""

fused_ext = load_inline(
    name='fused_post_conv_ext',
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
    # Perform the convolution with custom CUDA kernel
    # Note: This implementation assumes groups=1 and dilation=1 for simplicity
    # as the full generalized version would be significantly more complex
    out_channels = conv_transpose_weight.size(1)
    kernel_size = conv_transpose_weight.size(2)
    
    # Calculate output dimensions
    in_depth, in_height, in_width = x.shape[2], x.shape[3], x.shape[4]
    out_depth = (in_depth - 1) * conv_transpose_stride - 2 * conv_transpose_padding + kernel_size + conv_transpose_output_padding
    out_height = (in_height - 1) * conv_transpose_stride - 2 * conv_transpose_padding + kernel_size + conv_transpose_output_padding
    out_width = (in_width - 1) * conv_transpose_stride - 2 * conv_transpose_padding + kernel_size + conv_transpose_output_padding
    
    output_shape = (x.shape[0], out_channels, out_depth, out_height, out_width)
    x_conv = torch.empty(output_shape, device=x.device, dtype=x.dtype)
    
    fused_ext.conv_transpose3d(
        x, 
        conv_transpose_weight, 
        conv_transpose_bias, 
        x_conv,
        conv_transpose_stride,
        conv_transpose_padding,
        conv_transpose_output_padding
    )
    
    # Flatten bias for simplified kernel indexing
    bias_flat = bias.view(-1)
    
    # Use optimized fused kernel for the intensive post-processing element-wise ops
    return fused_ext.fused_post_conv(x_conv, bias_flat)

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
