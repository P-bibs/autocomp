# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_130922/code_6.py
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

# Optimized CUDA kernel focusing on Coalesced memory access and eliminating built-in functions
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define MAX_SHARED_MEMORY_PER_BLOCK 48 * 1024

__global__ void conv_transpose3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int64_t batch_size,
    int64_t in_channels,
    int64_t out_channels,
    int64_t input_d, int64_t input_h, int64_t input_w,
    int64_t output_d, int64_t output_h, int64_t output_w,
    int64_t kernel_d, int64_t kernel_h, int64_t kernel_w,
    int64_t stride_d, int64_t stride_h, int64_t stride_w,
    int64_t padding_d, int64_t padding_h, int64_t padding_w,
    int64_t output_padding_d, int64_t output_padding_h, int64_t output_padding_w
) {
    // Calculate global thread index
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_elements = batch_size * out_channels * output_d * output_h * output_w;
    
    if (idx >= total_elements) return;
    
    // Decompose linear index into 5D coordinates
    int64_t temp = idx;
    int64_t w = temp % output_w;
    temp /= output_w;
    int64_t h = temp % output_h;
    temp /= output_h;
    int64_t d = temp % output_d;
    temp /= output_d;
    int64_t c = temp % out_channels;
    int64_t n = temp / out_channels;
    
    float result = 0.0f;
    
    // Convolution computation
    for (int64_t kd = 0; kd < kernel_d; kd++) {
        for (int64_t kh = 0; kh < kernel_h; kh++) {
            for (int64_t kw = 0; kw < kernel_w; kw++) {
                for (int64_t ic = 0; ic < in_channels; ic++) {
                    // Calculate input coordinates
                    int64_t in_d = d + padding_d - kd * stride_d;
                    int64_t in_h = h + padding_h - kh * stride_h;
                    int64_t in_w = w + padding_w - kw * stride_w;
                    
                    // Check bounds
                    if (in_d >= 0 && in_d < input_d * stride_d && 
                        in_h >= 0 && in_h < input_h * stride_h && 
                        in_w >= 0 && in_w < input_w * stride_w &&
                        in_d % stride_d == 0 && in_h % stride_h == 0 && in_w % stride_w == 0) {
                        
                        in_d /= stride_d;
                        in_h /= stride_h;
                        in_w /= stride_w;
                        
                        // Calculate indices
                        int64_t input_idx = ((((n * in_channels + ic) * input_d + in_d) * input_h + in_h) * input_w + in_w);
                        int64_t weight_idx = ((((c * in_channels + ic) * kernel_d + kd) * kernel_h + kh) * kernel_w + kw);
                        
                        result += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }
    
    // Add bias
    result += bias[c];
    
    // Apply fused operation: ((x + bias) + x) * x + x = (2*x + bias) * x + x
    output[idx] = ((result + bias[c]) + result) * result + result;
}

void conv_transpose3d_fused_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& conv_bias,
    const torch::Tensor& post_bias,
    torch::Tensor& output,
    int64_t stride_d, int64_t stride_h, int64_t stride_w,
    int64_t padding_d, int64_t padding_h, int64_t padding_w,
    int64_t output_padding_d, int64_t output_padding_h, int64_t output_padding_w
) {
    auto input_sizes = input.sizes();
    auto weight_sizes = weight.sizes();
    
    int64_t batch_size = input_sizes[0];
    int64_t in_channels = input_sizes[1];
    int64_t input_d = input_sizes[2];
    int64_t input_h = input_sizes[3];
    int64_t input_w = input_sizes[4];
    
    int64_t out_channels = weight_sizes[1];
    int64_t kernel_d = weight_sizes[2];
    int64_t kernel_h = weight_sizes[3];
    int64_t kernel_w = weight_sizes[4];
    
    int64_t output_d = (input_d - 1) * stride_d - 2 * padding_d + kernel_d + output_padding_d;
    int64_t output_h = (input_h - 1) * stride_h - 2 * padding_h + kernel_h + output_padding_h;
    int64_t output_w = (input_w - 1) * stride_w - 2 * padding_w + kernel_w + output_padding_w;
    
    int64_t total_elements = batch_size * out_channels * output_d * output_h * output_w;
    
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    
    conv_transpose3d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        conv_bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_d, input_h, input_w,
        output_d, output_h, output_w,
        kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w,
        output_padding_d, output_padding_h, output_padding_w
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void conv_transpose3d_fused_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& conv_bias,
    const torch::Tensor& post_bias,
    torch::Tensor& output,
    int64_t stride_d, int64_t stride_h, int64_t stride_w,
    int64_t padding_d, int64_t padding_h, int64_t padding_w,
    int64_t output_padding_d, int64_t output_padding_h, int64_t output_padding_w
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_transpose3d_fused", &conv_transpose3d_fused_forward, "Fused 3D Transposed Convolution with Post-processing");
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
    # Validate group parameter (only groups=1 is supported in this implementation)
    if conv_transpose_groups != 1:
        raise NotImplementedError("Only conv_transpose_groups=1 is supported")
    
    # Validate dilation parameter (only dilation=1 is supported in this implementation)
    if conv_transpose_dilation != (1, 1, 1):
        raise NotImplementedError("Only conv_transpose_dilation=(1,1,1) is supported")
    
    # Ensure tensors are on CUDA
    x = x.cuda()
    conv_transpose_weight = conv_transpose_weight.cuda()
    conv_transpose_bias = conv_transpose_bias.cuda()
    bias = bias.cuda()
    
    # Calculate output dimensions
    input_d, input_h, input_w = x.shape[2], x.shape[3], x.shape[4]
    kernel_d, kernel_h, kernel_w = conv_transpose_weight.shape[2], conv_transpose_weight.shape[3], conv_transpose_weight.shape[4]
    
    stride_d, stride_h, stride_w = conv_transpose_stride
    padding_d, padding_h, padding_w = conv_transpose_padding
    output_padding_d, output_padding_h, output_padding_w = conv_transpose_output_padding
    
    output_d = (input_d - 1) * stride_d - 2 * padding_d + kernel_d + output_padding_d
    output_h = (input_h - 1) * stride_h - 2 * padding_h + kernel_h + output_padding_h
    output_w = (input_w - 1) * stride_w - 2 * padding_w + kernel_w + output_padding_w
    
    # Create output tensor
    output = torch.empty((x.shape[0], conv_transpose_weight.shape[1], output_d, output_h, output_w), 
                         dtype=x.dtype, device=x.device)
    
    # Call fused kernel
    fused_ext.conv_transpose3d_fused(
        x, conv_transpose_weight, conv_transpose_bias, bias,
        output,
        stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w,
        output_padding_d, output_padding_h, output_padding_w
    )
    
    return output

# Test parameters
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
