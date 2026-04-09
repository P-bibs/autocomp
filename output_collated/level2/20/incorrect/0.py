# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_124936/code_1.py
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

# --- CUDA Kernel ---
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// CUDA kernel for 3D transposed convolution
__global__ void conv_transpose3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ conv_bias,
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
    int output_padding,
    int dilation
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * out_depth * out_height * out_width;
    
    if (out_idx >= total_elements) return;
    
    int w = out_idx % out_width;
    out_idx /= out_width;
    int h = out_idx % out_height;
    out_idx /= out_height;
    int d = out_idx % out_depth;
    out_idx /= out_depth;
    int out_ch = out_idx % out_channels;
    int batch = out_idx / out_channels;
    
    float sum = 0.0f;
    
    // Calculate corresponding input position
    for (int kd = 0; kd < kernel_size; kd++) {
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                int in_d = d + padding - kd * dilation;
                int in_h = h + padding - kh * dilation;
                int in_w = w + padding - kw * dilation;
                
                if (in_d % stride == 0 && in_h % stride == 0 && in_w % stride == 0) {
                    in_d /= stride;
                    in_h /= stride;
                    in_w /= stride;
                    
                    if (in_d >= 0 && in_d < in_depth &&
                        in_h >= 0 && in_h < in_height &&
                        in_w >= 0 && in_w < in_width) {
                        
                        for (int in_ch = 0; in_ch < in_channels; in_ch++) {
                            int input_idx = batch * (in_channels * in_depth * in_height * in_width) +
                                          in_ch * (in_depth * in_height * in_width) +
                                          in_d * (in_height * in_width) +
                                          in_h * in_width +
                                          in_w;
                                          
                            int weight_idx = out_ch * (in_channels * kernel_size * kernel_size * kernel_size) +
                                           in_ch * (kernel_size * kernel_size * kernel_size) +
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
    
    output[out_idx * (out_depth * out_height * out_width) + 
           d * (out_height * out_width) + 
           h * out_width + 
           w] = sum + conv_bias[out_ch];
}

// CUDA kernel for fused arithmetic operations
__global__ void fused_arithmetic_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ bias, 
    float* __restrict__ output, 
    int numel, 
    int bias_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        // Calculate bias offset (scalar broadcast per channel)
        int bias_idx = (idx / (numel / bias_size)) % bias_size;
        
        float x = input[idx];
        float b = bias[bias_idx];
        
        // Simplified: x = ((x + b) + x) * x + x = (2x + b) * x + x = 2x^2 + bx + x
        output[idx] = (2.0f * x + b) * x + x;
    }
}

void fused_conv_transpose3d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor conv_bias,
    torch::Tensor output,
    torch::Tensor bias,
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
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    
    // First perform conv transpose
    conv_transpose3d_kernel<<<blocks, threads>>>(
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
        output_padding,
        dilation
    );
    
    // Then apply fused arithmetic
    torch::Tensor temp = output.clone();
    fused_arithmetic_kernel<<<blocks, threads>>>(
        temp.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        total_elements,
        bias.size(0)
    );
}
"""

# --- C++ Binding ---
cpp_source = r"""
#include <torch/extension.h>

void fused_conv_transpose3d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor conv_bias,
    torch::Tensor output,
    torch::Tensor bias,
    int stride,
    int padding,
    int output_padding,
    int dilation
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_transpose3d_forward", &fused_conv_transpose3d_forward, "Fused 3D transposed convolution with arithmetic operations");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_conv_transpose3d',
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
    # Only support groups=1 for this implementation
    if conv_transpose_groups != 1:
        raise NotImplementedError("Only groups=1 is supported")
    
    # Calculate output dimensions
    in_depth, in_height, in_width = x.shape[2], x.shape[3], x.shape[4]
    kernel_size = conv_transpose_weight.shape[2]
    
    out_depth = (in_depth - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_dilation * (kernel_size - 1) + 1 + conv_transpose_output_padding
    out_height = (in_height - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_dilation * (kernel_size - 1) + 1 + conv_transpose_output_padding
    out_width = (in_width - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_dilation * (kernel_size - 1) + 1 + conv_transpose_output_padding
    
    out_channels = conv_transpose_weight.shape[0]
    batch_size = x.shape[0]
    
    # Pre-allocate output
    output = torch.empty(batch_size, out_channels, out_depth, out_height, out_width, device=x.device, dtype=x.dtype)
    
    # Apply fused operation
    fused_ext.fused_conv_transpose3d_forward(
        x, 
        conv_transpose_weight, 
        conv_transpose_bias, 
        output, 
        bias.view(-1),
        conv_transpose_stride,
        conv_transpose_padding,
        conv_transpose_output_padding,
        conv_transpose_dilation
    )
    
    return output

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
    return [torch.rand(batch_size, in_channels, depth, height, width)]
