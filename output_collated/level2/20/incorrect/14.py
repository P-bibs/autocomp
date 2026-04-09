# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_130013/code_8.py
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# CUDA kernel source code for fused conv transpose 3d + element-wise operations
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

// Convolution transpose 3D kernel
__global__ void conv_transpose3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_d, int input_h, int input_w,
    int output_d, int output_h, int output_w,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int output_padding_d, int output_padding_h, int output_padding_w,
    int dilation_d, int dilation_h, int dilation_w,
    int groups
) {
    int out_elem = blockIdx.x * blockDim.x + threadIdx.x;
    int total_out_elems = batch_size * out_channels * output_d * output_h * output_w;
    
    if (out_elem >= total_out_elems) return;
    
    int out_w_idx = out_elem % output_w;
    out_elem /= output_w;
    int out_h_idx = out_elem % output_h;
    out_elem /= output_h;
    int out_d_idx = out_elem % output_d;
    out_elem /= output_d;
    int out_c_idx = out_elem % out_channels;
    int batch_idx = out_elem / out_channels;
    
    float sum = 0.0f;
    int group_idx = out_c_idx * groups / out_channels;
    
    for (int in_c = group_idx * in_channels / groups; in_c < (group_idx + 1) * in_channels / groups; in_c++) {
        for (int kd = 0; kd < kernel_d; kd++) {
            for (int kh = 0; kh < kernel_h; kh++) {
                for (int kw = 0; kw < kernel_w; kw++) {
                    int in_d = out_d_idx - kd * dilation_d + 2 * padding_d - output_padding_d;
                    int in_h = out_h_idx - kh * dilation_h + 2 * padding_h - output_padding_h;
                    int in_w = out_w_idx - kw * dilation_w + 2 * padding_w - output_padding_w;
                    
                    if (in_d % stride_d == 0 && in_h % stride_h == 0 && in_w % stride_w == 0) {
                        in_d /= stride_d;
                        in_h /= stride_h;
                        in_w /= stride_w;
                        
                        if (in_d >= 0 && in_d < input_d && in_h >= 0 && in_h < input_h && in_w >= 0 && in_w < input_w) {
                            int input_idx = batch_idx * (in_channels * input_d * input_h * input_w) +
                                            in_c * (input_d * input_h * input_w) +
                                            in_d * (input_h * input_w) +
                                            in_h * input_w +
                                            in_w;
                            
                            int weight_idx = out_c_idx * (in_channels / groups * kernel_d * kernel_h * kernel_w) +
                                             (in_c - group_idx * in_channels / groups) * (kernel_d * kernel_h * kernel_w) +
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
    
    if (bias != nullptr) {
        sum += bias[out_c_idx];
    }
    
    output[out_elem * (output_d * output_h * output_w) + 
           out_d_idx * (output_h * output_w) + 
           out_h_idx * output_w + 
           out_w_idx] = sum;
}

// Fused element-wise operations kernel
__global__ void fused_element_ops_kernel(
    const float* __restrict__ x,
    const float* __restrict__ additional_bias,
    float* __restrict__ output,
    int total_elements,
    int channels,
    int spatial_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_elements) {
        // Calculate channel index for bias lookup
        int channel_idx = (idx / spatial_elements) % channels;
        
        // Load input value
        float y = x[idx];
        float bias_val = additional_bias[channel_idx];
        
        // Fused operation: output = 2*y^2 + (bias_val + 1)*y
        float result = 2.0f * y * y + (bias_val + 1.0f) * y;
        
        output[idx] = result;
    }
}

void fused_conv_transpose3d_elementwise_ops(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor conv_bias,
    torch::Tensor additional_bias,
    torch::Tensor output,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int output_padding_d, int output_padding_h, int output_padding_w,
    int dilation_d, int dilation_h, int dilation_w,
    int groups
) {
    // Convolution transpose dimensions
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_d = input.size(2);
    int input_h = input.size(3);
    int input_w = input.size(4);
    
    int out_channels = weight.size(0);
    int kernel_d = weight.size(2);
    int kernel_h = weight.size(3);
    int kernel_w = weight.size(4);
    
    // Calculate output dimensions
    int output_d = (input_d - 1) * stride_d - 2 * padding_d + dilation_d * (kernel_d - 1) + output_padding_d + 1;
    int output_h = (input_h - 1) * stride_h - 2 * padding_h + dilation_h * (kernel_h - 1) + output_padding_h + 1;
    int output_w = (input_w - 1) * stride_w - 2 * padding_w + dilation_w * (kernel_w - 1) + output_padding_w + 1;
    
    // Temporary buffer for conv output
    auto conv_output = torch::empty({batch_size, out_channels, output_d, output_h, output_w}, 
                                    torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    
    // Launch conv transpose kernel
    int total_conv_output_elements = batch_size * out_channels * output_d * output_h * output_w;
    int threads = 256;
    int blocks = (total_conv_output_elements + threads - 1) / threads;
    
    conv_transpose3d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        conv_bias.defined() ? conv_bias.data_ptr<float>() : nullptr,
        conv_output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_d, input_h, input_w,
        output_d, output_h, output_w,
        kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w,
        output_padding_d, output_padding_h, output_padding_w,
        dilation_d, dilation_h, dilation_w,
        groups
    );
    
    // Launch fused element-wise operations kernel
    int total_elements = batch_size * out_channels * output_d * output_h * output_w;
    blocks = (total_elements + threads - 1) / threads;
    
    fused_element_ops_kernel<<<blocks, threads>>>(
        conv_output.data_ptr<float>(),
        additional_bias.data_ptr<float>(),
        output.data_ptr<float>(),
        total_elements,
        out_channels,
        output_d * output_h * output_w
    );
}
"""

# C++ binding code
cpp_source = r"""
#include <torch/extension.h>

void fused_conv_transpose3d_elementwise_ops(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor conv_bias,
    torch::Tensor additional_bias,
    torch::Tensor output,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int output_padding_d, int output_padding_h, int output_padding_w,
    int dilation_d, int dilation_h, int dilation_w,
    int groups
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_transpose3d_elementwise_ops", &fused_conv_transpose3d_elementwise_ops, "Fused conv transpose 3D and element-wise operations");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_conv_transpose3d_ops',
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
    # Ensure all tensors are on CUDA
    x = x.cuda()
    conv_transpose_weight = conv_transpose_weight.cuda()
    conv_transpose_bias = conv_transpose_bias.cuda() if conv_transpose_bias is not None else None
    bias = bias.cuda()
    
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
    
    # Create output tensor with correct shape
    batch_size = x.shape[0]
    out_channels = conv_transpose_weight.shape[0]
    
    # Calculate output dimensions
    input_d, input_h, input_w = x.shape[2], x.shape[3], x.shape[4]
    kernel_d, kernel_h, kernel_w = conv_transpose_weight.shape[2], conv_transpose_weight.shape[3], conv_transpose_weight.shape[4]
    
    output_d = (input_d - 1) * stride_d - 2 * padding_d + dilation_d * (kernel_d - 1) + output_padding_d + 1
    output_h = (input_h - 1) * stride_h - 2 * padding_h + dilation_h * (kernel_h - 1) + output_padding_h + 1
    output_w = (input_w - 1) * stride_w - 2 * padding_w + dilation_w * (kernel_w - 1) + output_padding_w + 1
    
    output = torch.empty((batch_size, out_channels, output_d, output_h, output_w), 
                        dtype=x.dtype, device=x.device)
    
    # Call fused kernel
    fused_ext.fused_conv_transpose3d_elementwise_ops(
        x.contiguous(),
        conv_transpose_weight.contiguous(),
        conv_transpose_bias.contiguous() if conv_transpose_bias is not None else torch.empty(0, device=x.device),
        bias.squeeze().contiguous(),
        output,
        stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w,
        output_padding_d, output_padding_h, output_padding_w,
        dilation_d, dilation_h, dilation_w,
        conv_transpose_groups
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
