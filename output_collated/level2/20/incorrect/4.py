# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_125325/code_3.py
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

# CUDA kernel: Fused Transposed Conv3D + Element-wise operations
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define CUDA_NUM_THREADS 256

__global__ void fused_conv_transpose3d_elementwise_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_depth,
    int input_height,
    int input_width,
    int output_depth,
    int output_height,
    int output_width,
    int kernel_size_d,
    int kernel_size_h,
    int kernel_size_w,
    int stride_d,
    int stride_h,
    int stride_w,
    int padding_d,
    int padding_h,
    int padding_w,
    int output_padding_d,
    int output_padding_h,
    int output_padding_w,
    int dilation_d,
    int dilation_h,
    int dilation_w
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * output_depth * output_height * output_width;
    
    if (idx >= total_elements) return;
    
    // Calculate indices
    int tmp = idx;
    int ow = tmp % output_width; tmp /= output_width;
    int oh = tmp % output_height; tmp /= output_height;
    int od = tmp % output_depth; tmp /= output_depth;
    int oc = tmp % out_channels; tmp /= out_channels;
    int b = tmp;
    
    float conv_out = 0.0f;
    
    // Iterate through kernel dimensions and input dimensions
    for (int kd = 0; kd < kernel_size_d; ++kd) {
        int d = od + padding_d - kd * dilation_d;
        if (d % stride_d != 0) continue;
        int id = d / stride_d;
        if (id < 0 || id >= input_depth) continue;
        
        for (int kh = 0; kh < kernel_size_h; ++kh) {
            int h = oh + padding_h - kh * dilation_h;
            if (h % stride_h != 0) continue;
            int ih = h / stride_h;
            if (ih < 0 || ih >= input_height) continue;
            
            for (int kw = 0; kw < kernel_size_w; ++kw) {
                int w = ow + padding_w - kw * dilation_w;
                if (w % stride_w != 0) continue;
                int iw = w / stride_w;
                if (iw < 0 || iw >= input_width) continue;
                
                for (int ic = 0; ic < in_channels; ++ic) {
                    int input_idx = ((((b * in_channels + ic) * input_depth + id) * input_height + ih) * input_width + iw);
                    int weight_idx = ((((oc * in_channels + ic) * kernel_size_d + kd) * kernel_size_h + kh) * kernel_size_w + kw);
                    conv_out += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    // Add bias from conv_transpose_bias
    conv_out += conv_bias[oc];
    
    // Perform element-wise operations
    float original_x = conv_out;
    float x = original_x + bias[oc]; // bias is per channel
    x = x + original_x;
    x = x * original_x;
    x = x + original_x;
    
    output[idx] = x;
}

void fused_conv_transpose3d_elementwise_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor conv_bias,
    torch::Tensor bias,
    torch::Tensor output,
    int stride_d,
    int stride_h,
    int stride_w,
    int padding_d,
    int padding_h,
    int padding_w,
    int output_padding_d,
    int output_padding_h,
    int output_padding_w,
    int dilation_d,
    int dilation_h,
    int dilation_w
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_depth = input.size(2);
    int input_height = input.size(3);
    int input_width = input.size(4);
    
    int out_channels = weight.size(1);
    int kernel_size_d = weight.size(2);
    int kernel_size_h = weight.size(3);
    int kernel_size_w = weight.size(4);
    
    int output_depth = output.size(2);
    int output_height = output.size(3);
    int output_width = output.size(4);
    
    int total_elements = batch_size * out_channels * output_depth * output_height * output_width;
    int blocks = (total_elements + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
    
    fused_conv_transpose3d_elementwise_kernel<<<blocks, CUDA_NUM_THREADS>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        conv_bias.data_ptr<float>(),
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
        kernel_size_d,
        kernel_size_h,
        kernel_size_w,
        stride_d,
        stride_h,
        stride_w,
        padding_d,
        padding_h,
        padding_w,
        output_padding_d,
        output_padding_h,
        output_padding_w,
        dilation_d,
        dilation_h,
        dilation_w
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_conv_transpose3d_elementwise_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor conv_bias,
    torch::Tensor bias,
    torch::Tensor output,
    int stride_d,
    int stride_h,
    int stride_w,
    int padding_d,
    int padding_h,
    int padding_w,
    int output_padding_d,
    int output_padding_h,
    int output_padding_w,
    int dilation_d,
    int dilation_h,
    int dilation_w
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_conv_transpose3d_elementwise_forward, "Fused ConvTranspose3D with elementwise operations");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_conv_transpose3d_elementwise',
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
    # Validate assumptions (groups must be 1 for this simplified implementation)
    assert conv_transpose_groups == 1, "This implementation only supports groups=1"
    
    # Handle stride, padding, output_padding, dilation tuples
    if isinstance(conv_transpose_stride, int):
        stride_d = stride_h = stride_w = conv_transpose_stride
    else:
        stride_d, stride_h, stride_w = conv_transpose_stride
        
    if isinstance(conv_transpose_padding, int):
        padding_d = padding_h = padding_w = conv_transpose_padding
    else:
        padding_d, padding_h, padding_w = conv_transpose_padding
        
    if isinstance(conv_transpose_output_padding, int):
        output_padding_d = output_padding_h = output_padding_w = conv_transpose_output_padding
    else:
        output_padding_d, output_padding_h, output_padding_w = conv_transpose_output_padding
        
    if isinstance(conv_transpose_dilation, int):
        dilation_d = dilation_h = dilation_w = conv_transpose_dilation
    else:
        dilation_d, dilation_h, dilation_w = conv_transpose_dilation

    # Compute output dimensions
    batch_size, in_channels, input_depth, input_height, input_width = x.shape
    out_channels, _, kernel_size_d, kernel_size_h, kernel_size_w = conv_transpose_weight.shape[:5]
    
    output_depth = (input_depth - 1) * stride_d - 2 * padding_d + dilation_d * (kernel_size_d - 1) + output_padding_d + 1
    output_height = (input_height - 1) * stride_h - 2 * padding_h + dilation_h * (kernel_size_h - 1) + output_padding_h + 1
    output_width = (input_width - 1) * stride_w - 2 * padding_w + dilation_w * (kernel_size_w - 1) + output_padding_w + 1
    
    # Create output tensor
    output = torch.empty((batch_size, out_channels, output_depth, output_height, output_width), device=x.device, dtype=x.dtype)
    
    # Call the fused operation
    fused_ext.fused_op(
        x, 
        conv_transpose_weight, 
        conv_transpose_bias, 
        bias.squeeze(),  # Remove singleton dimensions from bias
        output,
        stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w,
        output_padding_d, output_padding_h, output_padding_w,
        dilation_d, dilation_h, dilation_w
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
