# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_044115/code_7.py
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
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# ----------------------------------------------------------------------
# CUDA source – fused convolution transpose + addition + hardswish multiply
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>

// Hardswish activation function
__device__ __forceinline__ float hardswish(float x) {
    return x * fminf(fmaxf(x + 3.0f, 0.0f), 6.0f) / 6.0f;
}

// Custom 3D convolution transpose kernel
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
    int dilation_d, int dilation_h, int dilation_w,
    int groups
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
    int batch = temp / out_channels;
    
    float sum = 0.0f;
    
    // Determine which group this output channel belongs to
    int group = c_out * groups / out_channels;
    int weight_offset = group * (out_channels / groups) * in_channels * kernel_d * kernel_h * kernel_w;
    int in_channels_per_group = in_channels / groups;
    int out_channels_per_group = out_channels / groups;
    int c_out_in_group = c_out % out_channels_per_group;
    
    // Iterate through kernel dimensions
    for (int kd = 0; kd < kernel_d; kd++) {
        for (int kh = 0; kh < kernel_h; kh++) {
            for (int kw = 0; kw < kernel_w; kw++) {
                // Calculate corresponding input position
                int in_d = d_idx - kd * dilation_d + 2 * padding_d;
                int in_h = h_idx - kh * dilation_h + 2 * padding_h;
                int in_w = w_idx - kw * dilation_w + 2 * padding_w;
                
                // Check if divisible by stride
                if (in_d % stride_d == 0 && in_h % stride_h == 0 && in_w % stride_w == 0) {
                    in_d /= stride_d;
                    in_h /= stride_h;
                    in_w /= stride_w;
                    
                    // Check bounds
                    if (in_d >= 0 && in_d < input_d &&
                        in_h >= 0 && in_h < input_h &&
                        in_w >= 0 && in_w < input_w) {
                        
                        // Iterate through input channels in this group
                        for (int c_in_group = 0; c_in_group < in_channels_per_group; c_in_group++) {
                            int c_in = group * in_channels_per_group + c_in_group;
                            
                            // Calculate indices
                            int input_idx = batch * (in_channels * input_d * input_h * input_w) +
                                          c_in * (input_d * input_h * input_w) +
                                          in_d * (input_h * input_w) +
                                          in_h * input_w +
                                          in_w;
                                          
                            int weight_idx = weight_offset + 
                                           c_out_in_group * (in_channels_per_group * kernel_d * kernel_h * kernel_w) +
                                           c_in_group * (kernel_d * kernel_h * kernel_w) +
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
    
    output[out_idx] = sum;
}

// Fused kernel for addition and hardswish
__global__ void fused_add_hardswish_kernel(
    const float* __restrict__ conv_out,
    const float* __restrict__ add_input,
    float* __restrict__ output,
    const int N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < N; i += stride) {
        float val = conv_out[i] + add_input[i];
        output[i] = val * hardswish(val);
    }
}

void launch_conv_transpose3d(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int dilation_d, int dilation_h, int dilation_w,
    int groups) {
    
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
    
    const int block_size = 256;
    const int grid_size = min(65535, (total_elements + block_size - 1) / block_size);
    
    conv_transpose3d_kernel<<<grid_size, block_size>>>(
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
        groups
    );
}

void launch_fused_add_hardswish(
    torch::Tensor conv_out,
    torch::Tensor add_input,
    torch::Tensor output) {
    
    const int N = static_cast<int>(conv_out.numel());
    const int block_size = 256;
    const int grid_size = min(65535, (N + block_size - 1) / block_size);
    
    fused_add_hardswish_kernel<<<grid_size, block_size>>>(
        conv_out.data_ptr<float>(),
        add_input.data_ptr<float>(),
        output.data_ptr<float>(),
        N
    );
}
"""

# ----------------------------------------------------------------------
# C++ binding – exposes the fused functions to Python
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void launch_conv_transpose3d(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int dilation_d, int dilation_h, int dilation_w,
    int groups);

void launch_fused_add_hardswish(
    torch::Tensor conv_out,
    torch::Tensor add_input,
    torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_conv_transpose3d", &launch_conv_transpose3d, "Custom ConvTranspose3D CUDA kernel");
    m.def("launch_fused_add_hardswish", &launch_fused_add_hardswish, "Fused Add + HardSwish CUDA kernel");
}
"""

# ----------------------------------------------------------------------
# Compile the inline CUDA extension
# ----------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# ----------------------------------------------------------------------
# Functional model – uses custom CUDA kernels
# ----------------------------------------------------------------------
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
    # Calculate output dimensions for conv transpose
    batch_size, in_channels, D, H, W = x.shape
    out_channels = conv_transpose_weight.shape[1]
    kernel_d, kernel_h, kernel_w = conv_transpose_weight.shape[2], conv_transpose_weight.shape[3], conv_transpose_weight.shape[4]
    
    # Handle stride, padding, and dilation
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
    
    # Calculate output dimensions
    out_d = (D - 1) * stride_d - 2 * padding_d + dilation_d * (kernel_d - 1) + 1 + conv_transpose_output_padding
    out_h = (H - 1) * stride_h - 2 * padding_h + dilation_h * (kernel_h - 1) + 1 + conv_transpose_output_padding
    out_w = (W - 1) * stride_w - 2 * padding_w + dilation_w * (kernel_w - 1) + 1 + conv_transpose_output_padding
    
    # Create output tensor for conv transpose
    conv_output = torch.empty((batch_size, out_channels, out_d, out_h, out_w), device=x.device, dtype=x.dtype)
    
    # Launch custom conv transpose kernel
    fused_ext.launch_conv_transpose3d(
        x.contiguous(),
        conv_transpose_weight.contiguous(),
        conv_transpose_bias.contiguous(),
        conv_output,
        stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w,
        dilation_d, dilation_h, dilation_w,
        conv_transpose_groups
    )
    
    # Create final output tensor
    final_output = torch.empty_like(conv_output)
    
    # Launch fused add + hardswish kernel
    fused_ext.launch_fused_add_hardswish(
        conv_output.contiguous(),
        add_input.contiguous(),
        final_output
    )
    
    return final_output


# ----------------------------------------------------------------------
# Helper functions required by the test harness
# ----------------------------------------------------------------------
batch_size = 128
in_channels = 32
out_channels = 64
D = H = W = 16
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1, 1, 1)

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape]

def get_inputs():
    return [
        torch.rand(batch_size, in_channels, D, H, W),
        torch.rand(batch_size, out_channels, D * stride, H * stride, W * stride),
    ]
