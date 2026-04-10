# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_115141/code_1.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'min_value', 'divisor']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'min_value', 'divisor']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias']


class ModelNew(nn.Module):
    """
    A model that performs a transposed 3D convolution, clamps the output to a minimum value, 
    and then divides the result by a constant.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, min_value, divisor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.min_value = min_value
        self.divisor = divisor

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
    if 'min_value' in flat_state:
        state_kwargs['min_value'] = flat_state['min_value']
    else:
        state_kwargs['min_value'] = getattr(model, 'min_value')
    if 'divisor' in flat_state:
        state_kwargs['divisor'] = flat_state['divisor']
    else:
        state_kwargs['divisor'] = getattr(model, 'divisor')
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

# Optimization: Merge low-level operations - Implementing a custom fused 3D transposed convolution with clamp and division
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

// CUDA kernel for fused 3D transposed convolution + clamp + division
__global__ void fused_conv_transpose3d_clamp_div_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_d, int input_h, int input_w,
    int output_d, int output_h, int output_w,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int dilation_d, int dilation_h, int dilation_w,
    int groups,
    float min_value,
    float divisor
) {
    // Compute output index
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output_elements = batch_size * out_channels * output_d * output_h * output_w;
    
    if (out_idx >= total_output_elements) return;
    
    // Decompose output index
    int temp = out_idx;
    int w_out = temp % output_w;
    temp /= output_w;
    int h_out = temp % output_h;
    temp /= output_h;
    int d_out = temp % output_d;
    temp /= output_d;
    int c_out = temp % out_channels;
    int n = temp / out_channels;
    
    float acc = 0.0f;
    
    // For transposed convolution, we iterate over input positions that contribute to this output
    int group_id = c_out / (out_channels / groups);
    
    // Compute output padding adjustments
    int base_d = d_out - dilation_d * (kernel_d - 1) - padding_d;
    int base_h = h_out - dilation_h * (kernel_h - 1) - padding_h;
    int base_w = w_out - dilation_w * (kernel_w - 1) - padding_w;
    
    for (int kd = 0; kd < kernel_d; kd++) {
        for (int kh = 0; kh < kernel_h; kh++) {
            for (int kw = 0; kw < kernel_w; kw++) {
                int input_d_pos = base_d + kd * dilation_d;
                int input_h_pos = base_h + kh * dilation_h;
                int input_w_pos = base_w + kw * dilation_w;
                
                if (input_d_pos % stride_d == 0 && input_h_pos % stride_h == 0 && input_w_pos % stride_w == 0) {
                    input_d_pos /= stride_d;
                    input_h_pos /= stride_h;
                    input_w_pos /= stride_w;
                    
                    if (input_d_pos >= 0 && input_d_pos < input_d &&
                        input_h_pos >= 0 && input_h_pos < input_h &&
                        input_w_pos >= 0 && input_w_pos < input_w) {
                        
                        for (int c_in_group = 0; c_in_group < in_channels / groups; c_in_group++) {
                            int c_in = group_id * (in_channels / groups) + c_in_group;
                            
                            int input_idx = ((((n * in_channels + c_in) * input_d + input_d_pos) * input_h + input_h_pos) * input_w + input_w_pos);
                            int weight_idx = ((((c_in_group * kernel_d + kd) * kernel_h + kh) * kernel_w + kw) * out_channels + c_out);
                            
                            acc += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }
    }
    
    // Add bias
    acc += bias[c_out];
    
    // Apply clamp and division
    if (acc < min_value) {
        acc = min_value;
    }
    acc = acc / divisor;
    
    output[out_idx] = acc;
}

void launch_fused_conv_transpose3d_clamp_div(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int dilation_d, int dilation_h, int dilation_w,
    int groups,
    float min_value,
    float divisor
) {
    // Set CUDA device guard
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    
    // Tensor dimensions
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto input_d = input.size(2);
    auto input_h = input.size(3);
    auto input_w = input.size(4);
    
    auto out_channels = weight.size(1);
    auto kernel_d = weight.size(2);
    auto kernel_h = weight.size(3);
    auto kernel_w = weight.size(4);
    
    auto output_d = output.size(2);
    auto output_h = output.size(3);
    auto output_w = output.size(4);
    
    // Launch configuration
    int total_output_elements = batch_size * out_channels * output_d * output_h * output_w;
    int threads = 256;
    int blocks = (total_output_elements + threads - 1) / threads;
    
    // Launch kernel
    fused_conv_transpose3d_clamp_div_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_d, input_h, input_w,
        output_d, output_h, output_w,
        kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w,
        dilation_d, dilation_h, dilation_w,
        groups,
        min_value,
        divisor
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void launch_fused_conv_transpose3d_clamp_div(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int dilation_d, int dilation_h, int dilation_w,
    int groups,
    float min_value,
    float divisor
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_transpose3d_clamp_div", &launch_fused_conv_transpose3d_clamp_div, "Fused 3D Transposed Convolution with Clamp and Division");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_conv_transpose3d_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
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
    min_value,
    divisor,
):
    # Handle stride, padding, dilation as tuples
    if isinstance(conv_transpose_stride, int):
        stride_d = stride_h = stride_w = conv_transpose_stride
    else:
        stride_d, stride_h, stride_w = conv_transpose_stride
    
    if isinstance(conv_transpose_padding, int):
        padding_d = padding_h = padding_w = conv_transpose_padding
    else:
        padding_d, padding_h, padding_w = conv_transpose_padding
        
    if isinstance(conv_transpose_dilation, int):
        dilation_d = dilation_h = dilation_w = conv_transpose_dilation
    else:
        dilation_d, dilation_h, dilation_w = conv_transpose_dilation
    
    # Calculate output dimensions for transposed convolution
    batch_size = x.shape[0]
    in_channels = x.shape[1]
    input_d = x.shape[2]
    input_h = x.shape[3]
    input_w = x.shape[4]
    
    out_channels = conv_transpose_weight.shape[1]
    
    # Output dimension calculations for transposed convolution
    output_d = (input_d - 1) * stride_d - 2 * padding_d + conv_transpose_weight.shape[2] + conv_transpose_output_padding
    output_h = (input_h - 1) * stride_h - 2 * padding_h + conv_transpose_weight.shape[3] + conv_transpose_output_padding
    output_w = (input_w - 1) * stride_w - 2 * padding_w + conv_transpose_weight.shape[4] + conv_transpose_output_padding
    
    # Create output tensor
    output = torch.empty((batch_size, out_channels, output_d, output_h, output_w), device=x.device, dtype=x.dtype)
    
    # Launch fused kernel
    fused_ext.fused_conv_transpose3d_clamp_div(
        x, conv_transpose_weight, conv_transpose_bias, output,
        stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w,
        dilation_d, dilation_h, dilation_w,
        conv_transpose_groups,
        float(min_value),
        float(divisor)
    )
    
    return output

batch_size = 16
in_channels = 64
out_channels = 128
depth, height, width = 24, 48, 48
kernel_size = 3
stride = 2
padding = 1
min_value = -1.0
divisor = 2.0

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, min_value, divisor]

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]
