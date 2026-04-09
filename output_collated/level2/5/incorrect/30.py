# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_115905/code_0.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'bias_shape', 'stride', 'padding', 'output_padding']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'bias']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a transposed convolution, subtracts a bias term, and applies tanh activation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, bias_shape, stride=2, padding=1, output_padding=1):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
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
    # State for conv_transpose (nn.ConvTranspose2d)
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

# CUDA kernel for fused bias subtraction and tanh activation
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_bias_tanh_kernel(float* __restrict__ data, const float* __restrict__ bias, 
                                       int N, int C, int H, int W) {
    int total_elements = N * C * H * W;
    int hw = H * W;
    
    // Grid-stride loop
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total_elements; i += blockDim.x * gridDim.x) {
        int n = i / (C * hw);
        int c = (i / hw) % C;
        
        float val = data[i];
        // Bias is shaped (C, 1, 1), broadcasted across H and W
        val = val - bias[c];
        data[i] = tanhf(val);
    }
}

// Optimized CUDA kernel for transposed convolution
__global__ void conv_transpose2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int in_height,
    int in_width,
    int out_channels,
    int kernel_size,
    int stride,
    int padding,
    int output_padding
) {
    int out_height = (in_height - 1) * stride - 2 * padding + kernel_size + output_padding;
    int out_width = (in_width - 1) * stride - 2 * padding + kernel_size + output_padding;
    int total_output_elements = batch_size * out_channels * out_height * out_width;
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_size = blockDim.x * gridDim.x;
    
    // Grid-stride loop
    for (int idx = tid; idx < total_output_elements; idx += stride_size) {
        int temp = idx;
        int out_w = temp % out_width;
        temp /= out_width;
        int out_h = temp % out_height;
        temp /= out_height;
        int out_c = temp % out_channels;
        int batch = temp / out_channels;
        
        float sum = 0.0f;
        
        // Calculate input region that contributes to this output pixel
        int in_h_start = max(0, (out_h + padding - kernel_size + 1 + stride - 1) / stride);
        int in_h_end = min(in_height, (out_h + padding + stride) / stride);
        int in_w_start = max(0, (out_w + padding - kernel_size + 1 + stride - 1) / stride);
        int in_w_end = min(in_width, (out_w + padding + stride) / stride);
        
        for (int in_c = 0; in_c < in_channels; ++in_c) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    int in_h = out_h + padding - kh;
                    int in_w = out_w + padding - kw;
                    
                    if (in_h % stride == 0 && in_w % stride == 0) {
                        in_h /= stride;
                        in_w /= stride;
                        
                        if (in_h >= 0 && in_h < in_height && in_w >= 0 && in_w < in_width) {
                            int input_idx = batch * (in_channels * in_height * in_width) + 
                                          in_c * (in_height * in_width) + 
                                          in_h * in_width + in_w;
                            int weight_idx = out_c * (in_channels * kernel_size * kernel_size) + 
                                           in_c * (kernel_size * kernel_size) + 
                                           kh * kernel_size + kw;
                            sum += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
        
        // Add bias
        sum += bias[out_c];
        output[idx] = sum;
    }
}

void fused_op_forward(torch::Tensor x, torch::Tensor bias) {
    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    
    int total_elements = N * C * H * W;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    // Cap blocks to avoid excessive resource usage
    blocks = std::min(blocks, 65535);

    fused_bias_tanh_kernel<<<blocks, threads>>>(x.data_ptr<float>(), bias.data_ptr<float>(), N, C, H, W);
}

void conv_transpose2d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int stride,
    int padding,
    int output_padding
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_height = input.size(2);
    int in_width = input.size(3);
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);
    
    int out_height = (in_height - 1) * stride - 2 * padding + kernel_size + output_padding;
    int out_width = (in_width - 1) * stride - 2 * padding + kernel_size + output_padding;
    
    int total_output_elements = batch_size * out_channels * out_height * out_width;
    int threads = 256;
    int blocks = (total_output_elements + threads - 1) / threads;
    blocks = std::min(blocks, 65535);
    
    conv_transpose2d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        in_height,
        in_width,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(torch::Tensor x, torch::Tensor bias);
void conv_transpose2d_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output, int stride, int padding, int output_padding);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused Bias Subtraction and Tanh");
    m.def("conv_transpose2d", &conv_transpose2d_forward, "Conv Transpose 2D");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_op',
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
):
    # 1. Custom CUDA implementation of conv transpose 2d
    # Calculate output dimensions
    batch_size = x.size(0)
    in_channels = x.size(1)
    in_height = x.size(2)
    in_width = x.size(3)
    out_channels = conv_transpose_weight.size(0)
    kernel_size = conv_transpose_weight.size(2)
    
    # Only support group=1 and dilation=1 for this implementation
    assert conv_transpose_groups == 1, "Only groups=1 is supported"
    assert conv_transpose_dilation == 1, "Only dilation=1 is supported"
    
    out_height = (in_height - 1) * conv_transpose_stride - 2 * conv_transpose_padding + kernel_size + conv_transpose_output_padding
    out_width = (in_width - 1) * conv_transpose_stride - 2 * conv_transpose_padding + kernel_size + conv_transpose_output_padding
    
    # Create output tensor
    output = torch.empty((batch_size, out_channels, out_height, out_width), device=x.device, dtype=x.dtype)
    
    # Perform convolution using custom CUDA kernel
    fused_ext.conv_transpose2d(
        x, 
        conv_transpose_weight, 
        conv_transpose_bias,
        output,
        conv_transpose_stride,
        conv_transpose_padding,
        conv_transpose_output_padding
    )
    
    # 2. Fused operation using our custom CUDA kernel
    # Convert bias (C, 1, 1) to contiguous 1D for the kernel
    bias_shape = (out_channels, 1, 1)
    bias = torch.zeros(bias_shape, device=x.device, dtype=x.dtype)
    bias_flat = bias.view(-1).contiguous()
    output = output.contiguous()
    fused_ext.fused_op(output, bias_flat)
    
    return output

# Placeholder parameters as defined in the prompt
batch_size = 32
in_channels  = 64  
out_channels = 64  
height = width = 256 
kernel_size = 4
bias_shape = (out_channels, 1, 1)

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width, device='cuda')]
