# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_161645/code_0.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'output_padding', 'bias_shape', 'scaling_factor']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'bias', 'scaling_factor']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a transposed convolution, adds a bias term, clamps, scales, clamps, and divides.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor

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
    if 'scaling_factor' in flat_state:
        state_kwargs['scaling_factor'] = flat_state['scaling_factor']
    else:
        state_kwargs['scaling_factor'] = getattr(model, 'scaling_factor')
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

# CUDA kernel for fused operations
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_op_kernel(
    const float* __restrict__ input,
    const float* __restrict__ conv_weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ add_bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w,
    float scaling_factor
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * out_height * out_width;
    
    if (idx >= total_elements) return;
    
    // Calculate output coordinates
    int w_out = idx % out_width;
    int h_out = (idx / out_width) % out_height;
    int c_out = (idx / (out_width * out_height)) % out_channels;
    int n = idx / (out_width * out_height * out_channels);
    
    // Compute transposed convolution
    float sum = (conv_bias != nullptr) ? conv_bias[c_out] : 0.0f;
    
    // Loop over kernel elements
    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                // Calculate corresponding input position
                int h_in = h_out - padding_h + kh * dilation_h;
                int w_in = w_out - padding_w + kw * dilation_w;
                
                // Check if within input bounds and stride alignment
                if (h_in % stride_h == 0 && w_in % stride_w == 0) {
                    h_in /= stride_h;
                    w_in /= stride_w;
                    
                    if (h_in >= 0 && h_in < in_height && w_in >= 0 && w_in < in_width) {
                        int input_idx = ((n * in_channels + c_in) * in_height + h_in) * in_width + w_in;
                        int weight_idx = ((c_out * in_channels + c_in) * kernel_h + kh) * kernel_w + kw;
                        sum += input[input_idx] * conv_weight[weight_idx];
                    }
                }
            }
        }
    }
    
    // Add bias
    sum += add_bias[c_out];
    
    // First clamp [0, 1]
    sum = fmaxf(0.0f, fminf(1.0f, sum));
    
    // Scale
    sum *= scaling_factor;
    
    // Second clamp [0, 1]
    sum = fmaxf(0.0f, fminf(1.0f, sum));
    
    // Inverse scale
    sum /= scaling_factor;
    
    output[idx] = sum;
}

void fused_op_forward(
    const float* input,
    const float* conv_weight,
    const float* conv_bias,
    const float* add_bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w,
    float scaling_factor,
    int total_elements
) {
    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;
    
    fused_op_kernel<<<blocks, threads>>>(
        input, conv_weight, conv_bias, add_bias, output,
        batch_size, in_channels, out_channels, in_height, in_width,
        out_height, out_width, kernel_h, kernel_w,
        stride_h, stride_w, padding_h, padding_w,
        dilation_h, dilation_w, scaling_factor
    );
}
"""

# C++ binding code
cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(
    const float* input,
    const float* conv_weight,
    const float* conv_bias,
    const float* add_bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w,
    float scaling_factor,
    int total_elements
);

torch::Tensor fused_op(
    torch::Tensor input,
    torch::Tensor conv_weight,
    torch::Tensor conv_bias,
    torch::Tensor add_bias,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w,
    float scaling_factor
) {
    auto output = torch::empty({batch_size, out_channels, out_height, out_width}, 
                               torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    
    int total_elements = batch_size * out_channels * out_height * out_width;
    
    fused_op_forward(
        input.data_ptr<float>(),
        conv_weight.data_ptr<float>(),
        conv_bias.defined() ? conv_bias.data_ptr<float>() : nullptr,
        add_bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        in_height,
        in_width,
        out_height,
        out_width,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
        scaling_factor,
        total_elements
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op, "Fused transposed convolution with bias add and clamping");
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
    bias,
    scaling_factor,
):
    # Ensure tensors are on CUDA
    x = x.cuda()
    conv_transpose_weight = conv_transpose_weight.cuda()
    conv_transpose_bias = conv_transpose_bias.cuda() if conv_transpose_bias is not None else None
    bias = bias.cuda()
    
    # Get dimensions
    batch_size, in_channels, in_height, in_width = x.shape
    out_channels = conv_transpose_weight.shape[0]
    kernel_h, kernel_w = conv_transpose_weight.shape[2], conv_transpose_weight.shape[3]
    
    # Handle tuple strides/padding/dilation
    if isinstance(conv_transpose_stride, int):
        stride_h = stride_w = conv_transpose_stride
    else:
        stride_h, stride_w = conv_transpose_stride
    
    if isinstance(conv_transpose_padding, int):
        padding_h = padding_w = conv_transpose_padding
    else:
        padding_h, padding_w = conv_transpose_padding
        
    if isinstance(conv_transpose_dilation, int):
        dilation_h = dilation_w = conv_transpose_dilation
    else:
        dilation_h, dilation_w = conv_transpose_dilation
    
    # Calculate output dimensions for transposed convolution
    out_height = (in_height - 1) * stride_h - 2 * padding_h + dilation_h * (kernel_h - 1) + 1 + conv_transpose_output_padding
    out_width = (in_width - 1) * stride_w - 2 * padding_w + dilation_w * (kernel_w - 1) + 1 + conv_transpose_output_padding
    
    # Call fused operation
    result = fused_ext.fused_op(
        x.contiguous(),
        conv_transpose_weight.contiguous(),
        conv_transpose_bias.contiguous() if conv_transpose_bias is not None else torch.empty(0, device='cuda'),
        bias.contiguous(),
        batch_size,
        in_channels,
        out_channels,
        in_height,
        in_width,
        out_height,
        out_width,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
        float(scaling_factor)
    )
    
    return result

batch_size = 128
in_channels  = 64  
out_channels = 64  
height = width = 128 
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1)
scaling_factor = 2.0

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]
