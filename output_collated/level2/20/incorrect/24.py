# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_130922/code_5.py
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
import torch.nn.functional as F

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_post_conv_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int64_t num_elements,
    const int64_t spatial_size,
    const int64_t out_channels
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        // Correct channel index calculation: divide by spatial_size to find channel
        int64_t channel_idx = (idx / spatial_size) % out_channels;
        float val = input[idx];
        float b = bias[channel_idx];
        
        // Fused arithmetic: ((x + b) + x) * x + x
        output[idx] = ((val + b) + val) * val + val;
    }
}

__global__ void conv_transpose3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int64_t batch_size,
    const int64_t in_channels,
    const int64_t out_channels,
    const int64_t input_depth,
    const int64_t input_height,
    const int64_t input_width,
    const int64_t output_depth,
    const int64_t output_height,
    const int64_t output_width,
    const int64_t kernel_size,
    const int64_t stride,
    const int64_t padding,
    const int64_t output_padding
) {
    int64_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_elements = batch_size * out_channels * output_depth * output_height * output_width;
    
    if (out_idx >= total_elements) return;
    
    // Calculate output indices
    int64_t tmp = out_idx;
    int64_t w_out = tmp % output_width;
    tmp /= output_width;
    int64_t h_out = tmp % output_height;
    tmp /= output_height;
    int64_t d_out = tmp % output_depth;
    tmp /= output_depth;
    int64_t c_out = tmp % out_channels;
    int64_t n = tmp / out_channels;
    
    float sum = 0.0f;
    
    // Loop over input channels and kernel
    for (int64_t c_in = 0; c_in < in_channels; c_in++) {
        for (int64_t kd = 0; kd < kernel_size; kd++) {
            for (int64_t kh = 0; kh < kernel_size; kh++) {
                for (int64_t kw = 0; kw < kernel_size; kw++) {
                    // Calculate input position
                    int64_t d_in = d_out + padding - kd;
                    int64_t h_in = h_out + padding - kh;
                    int64_t w_in = w_out + padding - kw;
                    
                    // Check if within input bounds
                    if (d_in % stride == 0 && h_in % stride == 0 && w_in % stride == 0) {
                        d_in /= stride;
                        h_in /= stride;
                        w_in /= stride;
                        
                        if (d_in >= 0 && d_in < input_depth &&
                            h_in >= 0 && h_in < input_height &&
                            w_in >= 0 && w_in < input_width) {
                            
                            // Calculate indices
                            int64_t input_idx = ((((n * in_channels + c_in) * input_depth + d_in) * input_height + h_in) * input_width + w_in);
                            int64_t weight_idx = ((((c_out * in_channels + c_in) * kernel_size + kd) * kernel_size + kh) * kernel_size + kw);
                            
                            sum += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
    }
    
    output[out_idx] = sum + bias[c_out];
}

void fused_post_conv_forward(
    const torch::Tensor& input,
    const torch::Tensor& bias,
    torch::Tensor& output
) {
    int64_t num_elements = input.numel();
    int64_t spatial_size = input.size(2) * input.size(3) * input.size(4);
    int64_t out_channels = input.size(1);
    
    int threads_per_block = 256;
    int blocks = (num_elements + threads_per_block - 1) / threads_per_block;
    
    fused_post_conv_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        num_elements,
        spatial_size,
        out_channels
    );
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
    int64_t batch_size = input.size(0);
    int64_t in_channels = input.size(1);
    int64_t input_depth = input.size(2);
    int64_t input_height = input.size(3);
    int64_t input_width = input.size(4);
    
    int64_t out_channels = weight.size(1);
    int64_t kernel_size = weight.size(2);
    
    int64_t output_depth = (input_depth - 1) * stride - 2 * padding + kernel_size + output_padding;
    int64_t output_height = (input_height - 1) * stride - 2 * padding + kernel_size + output_padding;
    int64_t output_width = (input_width - 1) * stride - 2 * padding + kernel_size + output_padding;
    
    int threads_per_block = 256;
    int64_t total_elements = batch_size * out_channels * output_depth * output_height * output_width;
    int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    conv_transpose3d_kernel<<<blocks, threads_per_block>>>(
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

cpp_source = r"""
#include <torch/extension.h>

void fused_post_conv_forward(const torch::Tensor& input, const torch::Tensor& bias, torch::Tensor& output);
void conv_transpose3d_forward(const torch::Tensor& input, const torch::Tensor& weight, const torch::Tensor& bias, 
                              torch::Tensor& output, int64_t stride, int64_t padding, int64_t output_padding);

torch::Tensor fused_post_conv(const torch::Tensor& input, const torch::Tensor& bias) {
    auto output = torch::empty_like(input);
    fused_post_conv_forward(input, bias, output);
    return output;
}

torch::Tensor conv_transpose3d_custom(const torch::Tensor& input, const torch::Tensor& weight, const torch::Tensor& bias,
                                      int64_t stride, int64_t padding, int64_t output_padding) {
    // Calculate output dimensions
    int64_t batch_size = input.size(0);
    int64_t in_channels = input.size(1);
    int64_t input_depth = input.size(2);
    int64_t input_height = input.size(3);
    int64_t input_width = input.size(4);
    
    int64_t out_channels = weight.size(1);
    int64_t kernel_size = weight.size(2);
    
    int64_t output_depth = (input_depth - 1) * stride - 2 * padding + kernel_size + output_padding;
    int64_t output_height = (input_height - 1) * stride - 2 * padding + kernel_size + output_padding;
    int64_t output_width = (input_width - 1) * stride - 2 * padding + kernel_size + output_padding;
    
    auto output = torch::empty({batch_size, out_channels, output_depth, output_height, output_width}, 
                               torch::TensorOptions().dtype(input.dtype()).device(input.device()));
    
    conv_transpose3d_forward(input, weight, bias, output, stride, padding, output_padding);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_post_conv", &fused_post_conv, "Optimized fused post-conv kernel");
    m.def("conv_transpose3d_custom", &conv_transpose3d_custom, "Custom conv transpose 3d");
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
    # Check that groups is 1 and dilation is (1,1,1) as we don't handle those cases in our custom kernel
    if conv_transpose_groups != 1 or conv_transpose_dilation != (1, 1, 1):
        raise ValueError("Custom kernel only supports groups=1 and dilation=(1,1,1)")
    
    # Perform the convolution with our custom kernel
    x = fused_ext.conv_transpose3d_custom(
        x, 
        conv_transpose_weight, 
        conv_transpose_bias,
        conv_transpose_stride[0],  # Assuming all stride values are the same
        conv_transpose_padding[0],  # Assuming all padding values are the same
        conv_transpose_output_padding[0]  # Assuming all output_padding values are the same
    )
    
    # Use optimized fused kernel for the intensive post-processing element-wise ops
    return fused_ext.fused_post_conv(x, bias.view(-1))

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
