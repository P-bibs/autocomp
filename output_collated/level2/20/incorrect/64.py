# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_134740/code_12.py
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

# Optimized CUDA kernel using shared memory for bias, __ldg for read-only loads, and larger block size
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
    // Shared memory for bias: reduces global memory bandwidth pressure
    extern __shared__ float s_bias[];

    // Load bias into shared memory using __ldg for read-only cache
    for (int i = threadIdx.x; i < out_channels; i += blockDim.x) {
        s_bias[i] = __ldg(&bias[i]);
    }
    __syncthreads();

    // Grid-stride loop
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; 
         idx < num_elements_float4; 
         idx += blockDim.x * gridDim.x) {
        
        int64_t base_channel_idx = (idx * 4 / spatial_size) % out_channels;
        
        // Use __ldg for read-only cache when loading input
        float4 x_vec = __ldg(&input[idx]);
        
        // Use shared memory lookup
        float b = s_bias[base_channel_idx];
        
        // Apply arithmetic: ((x + b) + x) * x + x  => (2x + b) * x + x
        float4 result;
        result.x = ((x_vec.x + b) + x_vec.x) * x_vec.x + x_vec.x;
        result.y = ((x_vec.y + b) + x_vec.y) * x_vec.y + x_vec.y;
        result.z = ((x_vec.z + b) + x_vec.z) * x_vec.z + x_vec.z;
        result.w = ((x_vec.w + b) + x_vec.w) * x_vec.w + x_vec.w;
        
        output[idx] = result;
    }
}

// Custom conv transpose 3D kernel
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
    int output_padding,
    int dilation
) {
    // Calculate output dimensions
    int out_depth = (in_depth - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1 + output_padding;
    int out_height = (in_height - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1 + output_padding;
    int out_width = (in_width - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1 + output_padding;
    
    // Global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * out_depth * out_height * out_width;
    
    if (idx >= total_elements) return;
    
    // Calculate indices
    int w = idx % out_width;
    int h = (idx / out_width) % out_height;
    int d = (idx / (out_width * out_height)) % out_depth;
    int c = (idx / (out_width * out_height * out_depth)) % out_channels;
    int b = idx / (out_width * out_height * out_depth * out_channels);
    
    float sum = 0.0f;
    
    // Calculate corresponding input position
    int in_d = (d + padding - (kernel_size - 1) * dilation) / stride;
    int in_h = (h + padding - (kernel_size - 1) * dilation) / stride;
    int in_w = (w + padding - (kernel_size - 1) * dilation) / stride;
    
    // Perform convolution transpose operation
    for (int kd = 0; kd < kernel_size; kd++) {
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                int inp_d = d - kd * dilation + padding;
                int inp_h = h - kh * dilation + padding;
                int inp_w = w - kw * dilation + padding;
                
                if (inp_d % stride == 0 && inp_h % stride == 0 && inp_w % stride == 0) {
                    inp_d /= stride;
                    inp_h /= stride;
                    inp_w /= stride;
                    
                    if (inp_d >= 0 && inp_d < in_depth &&
                        inp_h >= 0 && inp_h < in_height &&
                        inp_w >= 0 && inp_w < in_width) {
                        
                        for (int ic = 0; ic < in_channels; ic++) {
                            int input_idx = b * (in_channels * in_depth * in_height * in_width) +
                                          ic * (in_depth * in_height * in_width) +
                                          inp_d * (in_height * in_width) +
                                          inp_h * in_width +
                                          inp_w;
                          
                            int weight_idx = c * (in_channels * kernel_size * kernel_size * kernel_size) +
                                           ic * (kernel_size * kernel_size * kernel_size) +
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
    
    // Add bias
    sum += bias[c];
    
    output[idx] = sum;
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
    
    // Increased block size to 512 for better occupancy
    int threads_per_block = 512;
    int blocks = (num_elements_float4 + threads_per_block - 1) / threads_per_block;
    
    // Calculate shared memory size
    size_t shared_mem_size = out_channels * sizeof(float);
    
    const float4* input_ptr = reinterpret_cast<const float4*>(input.data_ptr<float>());
    float4* output_ptr = reinterpret_cast<float4*>(output.data_ptr<float>());
    
    fused_post_conv_kernel<<<blocks, threads_per_block, shared_mem_size>>>(
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
    int output_padding,
    int dilation
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_depth = input.size(2);
    int in_height = input.size(3);
    int in_width = input.size(4);
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);
    
    int out_depth = (in_depth - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1 + output_padding;
    int out_height = (in_height - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1 + output_padding;
    int out_width = (in_width - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1 + output_padding;
    
    int total_elements = batch_size * out_channels * out_depth * out_height * out_width;
    
    int threads_per_block = 512;
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
        output_padding,
        dilation
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_post_conv_forward(const torch::Tensor& input, const torch::Tensor& bias, torch::Tensor& output);
void conv_transpose3d_forward(const torch::Tensor& input, const torch::Tensor& weight, const torch::Tensor& bias, 
                             torch::Tensor& output, int stride, int padding, int output_padding, int dilation);

torch::Tensor fused_post_conv(const torch::Tensor& input, const torch::Tensor& bias) {
    auto output = torch::empty_like(input);
    fused_post_conv_forward(input, bias, output);
    return output;
}

torch::Tensor conv_transpose3d_custom(const torch::Tensor& input, const torch::Tensor& weight, const torch::Tensor& bias,
                                     int stride, int padding, int output_padding, int dilation) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_depth = input.size(2);
    int in_height = input.size(3);
    int in_width = input.size(4);
    int kernel_size = weight.size(2);
    int out_channels = weight.size(0);
    
    int out_depth = (in_depth - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1 + output_padding;
    int out_height = (in_height - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1 + output_padding;
    int out_width = (in_width - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1 + output_padding;
    
    auto output = torch::empty({batch_size, out_channels, out_depth, out_height, out_width}, 
                              torch::TensorOptions().dtype(input.dtype()).device(input.device()));
    
    conv_transpose3d_forward(input, weight, bias, output, stride, padding, output_padding, dilation);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_post_conv", &fused_post_conv, "Fused post-conv arithmetic with shared memory bias");
    m.def("conv_transpose3d_custom", &conv_transpose3d_custom, "Custom conv transpose 3D");
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
    x = fused_ext.conv_transpose3d_custom(x, conv_transpose_weight, conv_transpose_bias, 
                                         conv_transpose_stride, conv_transpose_padding, 
                                         conv_transpose_output_padding, conv_transpose_dilation)
    
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
