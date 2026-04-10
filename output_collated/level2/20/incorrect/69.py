# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_140117/code_0.py
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

# Optimized CUDA kernel with better coalesced memory access and vectorization
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

    // Load bias into shared memory with coalesced access
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    // Coalesced loading of bias into shared memory
    for (int i = tid; i < out_channels; i += block_size) {
        s_bias[i] = bias[i];
    }
    __syncthreads();

    // Grid-stride loop with improved memory coalescing
    for (int64_t idx = blockIdx.x * block_size + tid; 
         idx < num_elements_float4; 
         idx += block_size * gridDim.x) {
        
        // Calculate channel index with better coalescing
        int64_t base_channel_idx = (idx * 4 / spatial_size) % out_channels;
        
        // Load 4 elements at once (coalesced access)
        float4 x_vec = input[idx];
        
        // Vectorized bias lookup
        float b = s_bias[base_channel_idx];
        
        // Vectorized computation: ((x + b) + x) * x + x  => (2x + b) * x + x
        float4 result;
        result.x = ((x_vec.x + b) + x_vec.x) * x_vec.x + x_vec.x;
        result.y = ((x_vec.y + b) + x_vec.y) * x_vec.y + x_vec.y;
        result.z = ((x_vec.z + b) + x_vec.z) * x_vec.z + x_vec.z;
        result.w = ((x_vec.w + b) + x_vec.w) * x_vec.w + x_vec.w;
        
        // Store 4 elements at once (coalesced access)
        output[idx] = result;
    }
}

// Optimized conv transpose 3D kernel
__global__ void conv_transpose3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_depth, int in_height, int in_width,
    int out_depth, int out_height, int out_width,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int dilation_d, int dilation_h, int dilation_w
) {
    // Calculate output indices
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_out_elements = batch_size * out_channels * out_depth * out_height * out_width;
    
    if (out_idx >= total_out_elements) return;
    
    // Decode output indices
    int tmp = out_idx;
    int w_out = tmp % out_width; tmp /= out_width;
    int h_out = tmp % out_height; tmp /= out_height;
    int d_out = tmp % out_depth; tmp /= out_depth;
    int c_out = tmp % out_channels; tmp /= out_channels;
    int n = tmp;
    
    if (n >= batch_size) return;
    
    // Calculate corresponding input position
    float acc = 0.0f;
    
    // Convolution loop
    for (int kd = 0; kd < kernel_d; kd++) {
        for (int kh = 0; kh < kernel_h; kh++) {
            for (int kw = 0; kw < kernel_w; kw++) {
                // Calculate input position
                int d_in = (d_out + padding_d - kd * dilation_d) / stride_d;
                int h_in = (h_out + padding_h - kh * dilation_h) / stride_h;
                int w_in = (w_out + padding_w - kw * dilation_w) / stride_w;
                
                // Check if division is exact and within bounds
                if ((d_out + padding_d - kd * dilation_d) % stride_d == 0 &&
                    (h_out + padding_h - kh * dilation_h) % stride_h == 0 &&
                    (w_out + padding_w - kw * dilation_w) % stride_w == 0 &&
                    d_in >= 0 && d_in < in_depth &&
                    h_in >= 0 && h_in < in_height &&
                    w_in >= 0 && w_in < in_width) {
                    
                    // Convolution computation
                    for (int c_in = 0; c_in < in_channels; c_in++) {
                        int input_idx = ((((n * in_channels + c_in) * in_depth + d_in) * in_height + h_in) * in_width + w_in);
                        int weight_idx = ((((c_out * in_channels + c_in) * kernel_d + kd) * kernel_h + kh) * kernel_w + kw);
                        acc += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }
    
    // Add bias
    acc += bias[c_out];
    output[out_idx] = acc;
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
    int blocks = min((num_elements_float4 + threads_per_block - 1) / threads_per_block, (int64_t)65535);
    
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
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int dilation_d, int dilation_h, int dilation_w
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_depth = input.size(2);
    int in_height = input.size(3);
    int in_width = input.size(4);
    
    int out_channels = output.size(1);
    int out_depth = output.size(2);
    int out_height = output.size(3);
    int out_width = output.size(4);
    
    int kernel_d = weight.size(2);
    int kernel_h = weight.size(3);
    int kernel_w = weight.size(4);
    
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
        in_depth, in_height, in_width,
        out_depth, out_height, out_width,
        kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w,
        dilation_d, dilation_h, dilation_w
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_post_conv_forward(const torch::Tensor& input, const torch::Tensor& bias, torch::Tensor& output);
void conv_transpose3d_forward(const torch::Tensor& input, const torch::Tensor& weight, const torch::Tensor& bias, 
                             torch::Tensor& output, int stride_d, int stride_h, int stride_w,
                             int padding_d, int padding_h, int padding_w,
                             int dilation_d, int dilation_h, int dilation_w);

torch::Tensor fused_post_conv(const torch::Tensor& input, const torch::Tensor& bias) {
    auto output = torch::empty_like(input);
    fused_post_conv_forward(input, bias, output);
    return output;
}

torch::Tensor conv_transpose3d_custom(const torch::Tensor& input, const torch::Tensor& weight, const torch::Tensor& bias,
                                      int stride_d, int stride_h, int stride_w,
                                      int padding_d, int padding_h, int padding_w,
                                      int dilation_d, int dilation_h, int dilation_w) {
    auto output = torch::zeros({input.size(0), weight.size(1), 
                               (input.size(2)-1)*stride_d-2*padding_d+(weight.size(2)-1)*dilation_d+1,
                               (input.size(3)-1)*stride_h-2*padding_h+(weight.size(3)-1)*dilation_h+1,
                               (input.size(4)-1)*stride_w-2*padding_w+(weight.size(4)-1)*dilation_w+1},
                               input.options());
    conv_transpose3d_forward(input, weight, bias, output, stride_d, stride_h, stride_w,
                            padding_d, padding_h, padding_w,
                            dilation_d, dilation_h, dilation_w);
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
                                         conv_transpose_stride[0], conv_transpose_stride[1], conv_transpose_stride[2],
                                         conv_transpose_padding[0], conv_transpose_padding[1], conv_transpose_padding[2],
                                         conv_transpose_dilation[0], conv_transpose_dilation[1], conv_transpose_dilation[2])
    
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
