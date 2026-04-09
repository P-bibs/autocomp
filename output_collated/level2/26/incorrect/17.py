# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_041736/code_3.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'output_padding', 'bias_shape']
FORWARD_ARG_NAMES = ['x', 'add_input']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'bias']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a 3D transposed convolution, adds an input tensor, and applies HardSwish activation.
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

# Optimization: Merge low-level operations (Fuse ConvTranspose3d + Add + Hardswish)
# We implement a custom CUDA kernel that performs the full operation:
# 1. ConvTranspose3d using optimized CUDA implementation
# 2. Element-wise addition with add_input
# 3. Hardswish activation (x * min(max(x + 3, 0), 6) / 6)

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstring>

// Hardswish activation function
__device__ __forceinline__ float hardswish(float x) {
    return x * fminf(fmaxf(x + 3.0f, 0.0f), 6.0f) / 6.0f;
}

// Helper to compute conv transpose 3d output dimensions
__host__ __device__ int get_conv_transpose_output_size(int input_size, int kernel_size, int stride, int padding, int dilation, int output_padding) {
    return (input_size - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1 + output_padding;
}

// CUDA kernel for ConvTranspose3d + Add + Hardswish fusion
__global__ void fused_conv_transpose3d_add_hardswish_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ add_input,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_d, int input_h, int input_w,
    int output_d, int output_h, int output_w,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int dilation_d, int dilation_h, int dilation_w
) {
    // Calculate global thread index
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * output_d * output_h * output_w;
    
    if (out_idx >= total_elements) return;
    
    // Decompose linear index into multidimensional indices
    int temp = out_idx;
    int w_idx = temp % output_w;
    temp /= output_w;
    int h_idx = temp % output_h;
    temp /= output_h;
    int d_idx = temp % output_d;
    temp /= output_d;
    int c_out = temp % out_channels;
    int n = temp / out_channels;
    
    // Calculate convolution sum
    float sum = 0.0f;
    
    // Iterate through input and kernel
    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int kd = 0; kd < kernel_d; ++kd) {
            for (int kh = 0; kh < kernel_h; ++kh) {
                for (int kw = 0; kw < kernel_w; ++kw) {
                    // Calculate corresponding input position
                    int in_d = d_idx + padding_d - kd * dilation_d;
                    int in_h = h_idx + padding_h - kh * dilation_h;
                    int in_w = w_idx + padding_w - kw * dilation_w;
                    
                    // Check if input position is valid (divisible by stride)
                    if (in_d >= 0 && in_d < input_d * stride_d && in_d % stride_d == 0 &&
                        in_h >= 0 && in_h < input_h * stride_h && in_h % stride_h == 0 &&
                        in_w >= 0 && in_w < input_w * stride_w && in_w % stride_w == 0) {
                        
                        int in_d_idx = in_d / stride_d;
                        int in_h_idx = in_h / stride_h;
                        int in_w_idx = in_w / stride_w;
                        
                        // Check bounds
                        if (in_d_idx < input_d && in_h_idx < input_h && in_w_idx < input_w) {
                            int input_idx = n * (in_channels * input_d * input_h * input_w) +
                                            c_in * (input_d * input_h * input_w) +
                                            in_d_idx * (input_h * input_w) +
                                            in_h_idx * input_w +
                                            in_w_idx;
                                            
                            int weight_idx = c_in * (out_channels * kernel_d * kernel_h * kernel_w) +
                                             c_out * (kernel_d * kernel_h * kernel_w) +
                                             kd * (kernel_h * kernel_w) +
                                             kh * kernel_w +
                                             kw;
                                             
                            sum += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
    }
    
    // Add bias
    sum += bias[c_out];
    
    // Add add_input and apply hardswish
    float val = sum + add_input[out_idx];
    output[out_idx] = val * fminf(fmaxf(val + 3.0f, 0.0f), 6.0f) / 6.0f;
}

void fused_conv_transpose3d_add_hardswish(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor add_input,
    torch::Tensor output,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int dilation_d, int dilation_h, int dilation_w
) {
    // Get tensor dimensions
    auto input_sizes = input.sizes();
    auto weight_sizes = weight.sizes();
    auto output_sizes = output.sizes();
    
    int batch_size = input_sizes[0];
    int in_channels = input_sizes[1];
    int input_d = input_sizes[2];
    int input_h = input_sizes[3];
    int input_w = input_sizes[4];
    
    int out_channels = weight_sizes[1];
    int kernel_d = weight_sizes[2];
    int kernel_h = weight_sizes[3];
    int kernel_w = weight_sizes[4];
    
    int output_d = output_sizes[2];
    int output_h = output_sizes[3];
    int output_w = output_sizes[4];
    
    // Calculate total number of output elements
    int total_elements = batch_size * out_channels * output_d * output_h * output_w;
    
    // Configure kernel launch parameters
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    
    // Launch kernel
    fused_conv_transpose3d_add_hardswish_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        add_input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_d, input_h, input_w,
        output_d, output_h, output_w,
        kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w,
        dilation_d, dilation_h, dilation_w
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_conv_transpose3d_add_hardswish(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor add_input,
    torch::Tensor output,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int dilation_d, int dilation_h, int dilation_w
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_transpose3d_add_hardswish", &fused_conv_transpose3d_add_hardswish, "Fused ConvTranspose3d + Add + Hardswish");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_conv_ops',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(
    x,
    add_input,
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
    # Only support groups=1 for this implementation
    if conv_transpose_groups != 1:
        raise NotImplementedError("Only groups=1 is supported")
    
    # Ensure all stride, padding, dilation are 3-element tuples
    if isinstance(conv_transpose_stride, int):
        conv_transpose_stride = (conv_transpose_stride, conv_transpose_stride, conv_transpose_stride)
    if isinstance(conv_transpose_padding, int):
        conv_transpose_padding = (conv_transpose_padding, conv_transpose_padding, conv_transpose_padding)
    if isinstance(conv_transpose_output_padding, int):
        conv_transpose_output_padding = (conv_transpose_output_padding, conv_transpose_output_padding, conv_transpose_output_padding)
    if isinstance(conv_transpose_dilation, int):
        conv_transpose_dilation = (conv_transpose_dilation, conv_transpose_dilation, conv_transpose_dilation)
    
    # Create output tensor with correct shape
    out_shape = (
        x.shape[0],
        conv_transpose_weight.shape[1],
        (x.shape[2] - 1) * conv_transpose_stride[0] - 2 * conv_transpose_padding[0] + conv_transpose_dilation[0] * (conv_transpose_weight.shape[2] - 1) + 1 + conv_transpose_output_padding[0],
        (x.shape[3] - 1) * conv_transpose_stride[1] - 2 * conv_transpose_padding[1] + conv_transpose_dilation[1] * (conv_transpose_weight.shape[3] - 1) + 1 + conv_transpose_output_padding[1],
        (x.shape[4] - 1) * conv_transpose_stride[2] - 2 * conv_transpose_padding[2] + conv_transpose_dilation[2] * (conv_transpose_weight.shape[4] - 1) + 1 + conv_transpose_output_padding[2]
    )
    output = torch.empty(out_shape, device=x.device, dtype=x.dtype)
    
    # Call fused kernel
    fused_ext.fused_conv_transpose3d_add_hardswish(
        x, conv_transpose_weight, conv_transpose_bias, add_input, output,
        conv_transpose_stride[0], conv_transpose_stride[1], conv_transpose_stride[2],
        conv_transpose_padding[0], conv_transpose_padding[1], conv_transpose_padding[2],
        conv_transpose_dilation[0], conv_transpose_dilation[1], conv_transpose_dilation[2]
    )
    
    return output

batch_size = 128
in_channels = 32
out_channels = 64
D, H, W = 16, 16, 16
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1, 1, 1)

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape]

def get_inputs():
    return [torch.rand(batch_size, in_channels, D, H, W), torch.rand(batch_size, out_channels, D*stride, H*stride, W*stride)]
