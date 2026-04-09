# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_045050/code_5.py
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

# Define custom CUDA kernels for both conv_transpose3d and fused element-wise operations
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

// HardSwish activation function
__device__ __forceinline__ float hardswish(float x) {
    return x * fminf(fmaxf(x + 3.0f, 0.0f), 6.0f) / 6.0f;
}

// ConvTranspose3D kernel implementation (simplified version)
__global__ void conv_transpose3d_kernel(
    const float* input, const float* weight, const float* bias,
    float* output,
    int batch_size, int in_channels, int out_channels,
    int input_d, int input_h, int input_w,
    int output_d, int output_h, int output_w,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int output_padding_d, int output_padding_h, int output_padding_w,
    int dilation_d, int dilation_h, int dilation_w
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * output_d * output_h * output_w;
    
    if (out_idx >= total_elements) return;
    
    int w = out_idx % output_w;
    out_idx /= output_w;
    int h = out_idx % output_h;
    out_idx /= output_h;
    int d = out_idx % output_d;
    out_idx /= output_d;
    int c_out = out_idx % out_channels;
    int batch = out_idx / out_channels;
    
    float value = 0.0f;
    
    // Compute convolution transpose
    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int kd = 0; kd < kernel_d; ++kd) {
            for (int kh = 0; kh < kernel_h; ++kh) {
                for (int kw = 0; kw < kernel_w; ++kw) {
                    int in_d = d - stride_d * kd + 2 * padding_d - output_padding_d;
                    int in_h = h - stride_h * kh + 2 * padding_h - output_padding_h;
                    int in_w = w - stride_w * kw + 2 * padding_w - output_padding_w;
                    
                    if (in_d % dilation_d == 0 && in_h % dilation_h == 0 && in_w % dilation_w == 0) {
                        in_d /= dilation_d;
                        in_h /= dilation_h;
                        in_w /= dilation_w;
                        
                        if (in_d >= 0 && in_d < input_d && 
                            in_h >= 0 && in_h < input_h && 
                            in_w >= 0 && in_w < input_w) {
                            
                            int input_idx = batch * (in_channels * input_d * input_h * input_w) +
                                            c_in * (input_d * input_h * input_w) +
                                            in_d * (input_h * input_w) +
                                            in_h * input_w +
                                            in_w;
                            
                            int weight_idx = c_in * (out_channels * kernel_d * kernel_h * kernel_w) +
                                             c_out * (kernel_d * kernel_h * kernel_w) +
                                             kd * (kernel_h * kernel_w) +
                                             kh * kernel_w +
                                             kw;
                            
                            value += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
    }
    
    // Add bias
    value += bias[c_out];
    
    // Store result
    output[out_idx] = value;
}

// Fused add + HardSwish kernel
__global__ void fused_add_hardswish_kernel(float* x, const float* add_input, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        float val = x[idx] + add_input[idx];
        x[idx] = val * hardswish(val);
    }
}

// Host function for conv_transpose3d
void conv_transpose3d_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int output_padding_d, int output_padding_h, int output_padding_w,
    int dilation_d, int dilation_h, int dilation_w
) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_d = input.size(2);
    int input_h = input.size(3);
    int input_w = input.size(4);
    
    int out_channels = weight.size(1);
    int kernel_d = weight.size(2);
    int kernel_h = weight.size(3);
    int kernel_w = weight.size(4);
    
    int output_d = output.size(2);
    int output_h = output.size(3);
    int output_w = output.size(4);
    
    int total_elements = batch_size * out_channels * output_d * output_h * output_w;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    
    conv_transpose3d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        input_d, input_h, input_w,
        output_d, output_h, output_w,
        kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w,
        output_padding_d, output_padding_h, output_padding_w,
        dilation_d, dilation_h, dilation_w
    );
}

// Host function for fused operation
void fused_add_hardswish_cuda(torch::Tensor x, torch::Tensor add_input) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
    
    int num_elements = x.numel();
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;
    
    fused_add_hardswish_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        add_input.data_ptr<float>(),
        num_elements
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void conv_transpose3d_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int output_padding_d, int output_padding_h, int output_padding_w,
    int dilation_d, int dilation_h, int dilation_w
);

void fused_add_hardswish_cuda(torch::Tensor x, torch::Tensor add_input);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_transpose3d", &conv_transpose3d_cuda, "ConvTranspose3D CUDA implementation");
    m.def("fused_add_hardswish", &fused_add_hardswish_cuda, "Fused Add + HardSwish CUDA implementation");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='custom_kernels',
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
    # Extract dimensions
    stride_d, stride_h, stride_w = conv_transpose_stride
    padding_d, padding_h, padding_w = conv_transpose_padding
    output_padding_d, output_padding_h, output_padding_w = conv_transpose_output_padding
    dilation_d, dilation_h, dilation_w = conv_transpose_dilation
    
    # Create output tensor
    batch_size, in_channels, input_d, input_h, input_w = x.shape
    out_channels = conv_transpose_weight.shape[1]
    
    output_d = (input_d - 1) * stride_d - 2 * padding_d + conv_transpose_weight.shape[2] + output_padding_d
    output_h = (input_h - 1) * stride_h - 2 * padding_h + conv_transpose_weight.shape[3] + output_padding_h
    output_w = (input_w - 1) * stride_w - 2 * padding_w + conv_transpose_weight.shape[4] + output_padding_w
    
    output = torch.empty((batch_size, out_channels, output_d, output_h, output_w), device=x.device, dtype=x.dtype)
    
    # Perform conv_transpose3d using custom CUDA kernel
    fused_ext.conv_transpose3d(
        x, conv_transpose_weight, conv_transpose_bias, output,
        stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w,
        output_padding_d, output_padding_h, output_padding_w,
        dilation_d, dilation_h, dilation_w
    )
    
    # Apply fused add + hardswish
    fused_ext.fused_add_hardswish(output, add_input)
    
    return output

# Constants for test setup
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
