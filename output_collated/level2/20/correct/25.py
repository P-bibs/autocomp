# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_133559/code_3.py
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

# Optimized fused CUDA kernel that combines conv_transpose3d and post-processing
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_conv_transpose_post_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ post_bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int input_depth,
    const int input_height,
    const int input_width,
    const int output_depth,
    const int output_height,
    const int output_width,
    const int kernel_size,
    const int stride,
    const int padding,
    const int output_padding
) {
    // Shared memory for post_bias to reduce global memory accesses
    extern __shared__ float post_bias_shared[];
    
    int tid = threadIdx.x;
    if (tid < out_channels) {
        post_bias_shared[tid] = post_bias[tid];
    }
    __syncthreads();

    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output_elements = batch_size * out_channels * output_depth * output_height * output_width;
    
    if (idx >= total_output_elements) return;

    // Decode output tensor coordinates
    int temp = idx;
    int w_out = temp % output_width;
    temp /= output_width;
    int h_out = temp % output_height;
    temp /= output_height;
    int d_out = temp % output_depth;
    temp /= output_depth;
    int c_out = temp % out_channels;
    int n = temp / out_channels;

    // Load bias value for this channel from shared memory
    float b = post_bias_shared[c_out];

    // Perform convolution transpose calculation
    float sum = 0.0f;
    
    // Iterate through kernel positions
    for (int kd = 0; kd < kernel_size; kd++) {
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                // Calculate corresponding input position
                int d_in = (d_out + padding - kd) / stride;
                int h_in = (h_out + padding - kh) / stride;
                int w_in = (w_out + padding - kw) / stride;
                
                // Check if the division was exact (valid convolution position)
                if ((d_out + padding - kd) % stride == 0 &&
                    (h_out + padding - kh) % stride == 0 &&
                    (w_out + padding - kw) % stride == 0) {
                    
                    // Check bounds
                    if (d_in >= 0 && d_in < input_depth &&
                        h_in >= 0 && h_in < input_height &&
                        w_in >= 0 && w_in < input_width) {
                        
                        // Accumulate over input channels
                        for (int c_in = 0; c_in < in_channels; c_in++) {
                            // Calculate weight index: [in_channel, out_channel, kd, kh, kw]
                            int weight_idx = ((c_in * out_channels + c_out) * kernel_size + kd) * kernel_size * kernel_size +
                                             kh * kernel_size + kw;
                            
                            // Calculate input index: [batch, in_channel, d_in, h_in, w_in]
                            int input_idx = ((n * in_channels + c_in) * input_depth + d_in) * input_height * input_width +
                                            h_in * input_width + w_in;
                            
                            sum += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
    }
    
    // Add convolution bias
    sum += conv_bias[c_out];

    // Apply post-processing: x * (2*x + bias + 1)
    const float two = 2.0f;
    const float one = 1.0f;
    float result = sum * (two * sum + b + one);

    // Write result
    output[idx] = result;
}

// Host function that launches the kernel
void fused_conv_transpose_post_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& conv_bias,
    const torch::Tensor& post_bias,
    torch::Tensor& output,
    const int stride,
    const int padding,
    const int output_padding
) {
    const int batch_size = static_cast<int>(input.size(0));
    const int in_channels = static_cast<int>(input.size(1));
    const int input_depth = static_cast<int>(input.size(2));
    const int input_height = static_cast<int>(input.size(3));
    const int input_width = static_cast<int>(input.size(4));
    
    const int out_channels = static_cast<int>(weight.size(1));  // Note: conv_transpose has weights [in, out/groups, ...]
    const int kernel_size = static_cast<int>(weight.size(2));
    
    // Calculate output dimensions for conv_transpose3d
    const int output_depth = (input_depth - 1) * stride - 2 * padding + kernel_size + output_padding;
    const int output_height = (input_height - 1) * stride - 2 * padding + kernel_size + output_padding;
    const int output_width = (input_width - 1) * stride - 2 * padding + kernel_size + output_padding;
    
    const int total_output_elements = batch_size * out_channels * output_depth * output_height * output_width;
    
    const int threads = 256;
    const int blocks = (total_output_elements + threads - 1) / threads;
    
    const int shared_mem = out_channels * sizeof(float);

    fused_conv_transpose_post_kernel<<<blocks, threads, shared_mem>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        conv_bias.data_ptr<float>(),
        post_bias.data_ptr<float>(),
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

cpp_source = r"""
#include <torch/extension.h>

void fused_conv_transpose_post_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& conv_bias,
    const torch::Tensor& post_bias,
    torch::Tensor& output,
    const int stride,
    const int padding,
    const int output_padding
);

torch::Tensor fused_conv_transpose_post(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& conv_bias,
    const torch::Tensor& post_bias,
    const int stride,
    const int padding,
    const int output_padding
) {
    // Calculate output dimensions
    const int batch_size = static_cast<int>(input.size(0));
    const int in_channels = static_cast<int>(input.size(1));
    const int input_depth = static_cast<int>(input.size(2));
    const int input_height = static_cast<int>(input.size(3));
    const int input_width = static_cast<int>(input.size(4));
    const int out_channels = static_cast<int>(weight.size(1));
    const int kernel_size = static_cast<int>(weight.size(2));
    
    const int output_depth = (input_depth - 1) * stride - 2 * padding + kernel_size + output_padding;
    const int output_height = (input_height - 1) * stride - 2 * padding + kernel_size + output_padding;
    const int output_width = (input_width - 1) * stride - 2 * padding + kernel_size + output_padding;
    
    auto output = torch::empty({batch_size, out_channels, output_depth, output_height, output_width}, 
                               torch::dtype(input.dtype()).device(input.device()));
    
    fused_conv_transpose_post_forward(input, weight, conv_bias, post_bias, output, stride, padding, output_padding);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_transpose_post", &fused_conv_transpose_post,
          "Fused conv transpose and post-processing kernel");
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
    # Skip the groups and dilation parameters for simplicity as they're not used in the original
    # Directly fuse convolution transpose and post-processing into a single kernel
    bias_flat = bias.view(-1)
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
