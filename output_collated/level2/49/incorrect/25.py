# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_094958/code_2.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'output_padding', 'bias']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'softmax_dim']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a 3D transposed convolution, applies Softmax and Sigmoid.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=bias)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

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
    # State for softmax (nn.Softmax)
    state_kwargs['softmax_dim'] = model.softmax.dim
    # State for sigmoid (nn.Sigmoid)
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

# Define CUDA kernel for fused conv_transpose3d + softmax + sigmoid
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// CUDA kernel for transposed convolution
__global__ void conv_transpose3d_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_d, int input_h, int input_w,
    int output_d, int output_h, int output_w,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    int dilation
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = batch_size * out_channels * output_d * output_h * output_w;
    
    if (out_idx >= total_outputs) return;
    
    // Decode output index
    int tmp = out_idx;
    int w_out = tmp % output_w;
    tmp /= output_w;
    int h_out = tmp % output_h;
    tmp /= output_h;
    int d_out = tmp % output_d;
    tmp /= output_d;
    int c_out = tmp % out_channels;
    int b = tmp / out_channels;
    
    float sum = 0.0f;
    
    // Iterate through kernel
    for (int kd = 0; kd < kernel_size; kd++) {
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                // Calculate input position
                int d_in = d_out + padding - kd * dilation;
                int h_in = h_out + padding - kh * dilation;
                int w_in = w_out + padding - kw * dilation;
                
                // Check if in valid input range
                if (d_in % stride == 0 && h_in % stride == 0 && w_in % stride == 0) {
                    d_in /= stride;
                    h_in /= stride;
                    w_in /= stride;
                    
                    if (d_in >= 0 && d_in < input_d &&
                        h_in >= 0 && h_in < input_h &&
                        w_in >= 0 && w_in < input_w) {
                        
                        // Calculate indices
                        int input_idx = ((((b * in_channels) + 0) * input_d + d_in) * input_h + h_in) * input_w + w_in;
                        int weight_idx = ((((c_out * in_channels) + 0) * kernel_size + kd) * kernel_size + kh) * kernel_size + kw;
                        
                        for (int c_in = 0; c_in < in_channels; c_in++) {
                            int in_idx = input_idx + c_in * input_d * input_h * input_w;
                            int w_idx = weight_idx + c_in * kernel_size * kernel_size * kernel_size;
                            sum += input[in_idx] * weight[w_idx];
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

// CUDA kernel for fused softmax + sigmoid
__global__ void fused_softmax_sigmoid_kernel(
    const float* input,
    float* output,
    int numel,
    int channel_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= numel) return;
    
    // Calculate which channel group this element belongs to
    int group_idx = idx / channel_size;
    int local_idx = idx % channel_size;
    
    // Softmax: find max for numerical stability
    float max_val = -1e9f;
    for (int i = 0; i < channel_size; i++) {
        int pos = group_idx * channel_size + i;
        max_val = fmaxf(max_val, input[pos]);
    }
    
    // Compute exp and sum for softmax
    float exp_sum = 0.0f;
    for (int i = 0; i < channel_size; i++) {
        int pos = group_idx * channel_size + i;
        exp_sum += expf(input[pos] - max_val);
    }
    
    // Apply softmax and then sigmoid to current element
    int pos = idx;
    float softmax_val = expf(input[pos] - max_val) / exp_sum;
    float sigmoid_val = 1.0f / (1.0f + expf(-softmax_val));
    
    output[pos] = sigmoid_val;
}

void fused_conv_transpose3d_softmax_sigmoid_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int stride,
    int padding,
    int output_padding,
    int dilation,
    int softmax_dim
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
    
    int out_channels = weight_sizes[0];
    int kernel_size = weight_sizes[2]; // Assuming cubic kernel
    
    int output_d = output_sizes[2];
    int output_h = output_sizes[3];
    int output_w = output_sizes[4];
    
    // Launch conv_transpose3d kernel
    int conv_threads = 256;
    int conv_numel = batch_size * out_channels * output_d * output_h * output_w;
    int conv_blocks = (conv_numel + conv_threads - 1) / conv_threads;
    
    conv_transpose3d_kernel<<<conv_blocks, conv_threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_d, input_h, input_w,
        output_d, output_h, output_w,
        kernel_size,
        stride,
        padding,
        output_padding,
        dilation
    );
    
    // Calculate channel size for softmax
    int channel_size = 1;
    for (int i = softmax_dim + 1; i < 5; i++) { // 5 dimensions: (batch, channels, D, H, W)
        channel_size *= output_sizes[i];
    }
    
    // Launch fused softmax + sigmoid kernel
    int fuse_threads = 256;
    int fuse_blocks = (conv_numel + fuse_threads - 1) / fuse_threads;
    
    fused_softmax_sigmoid_kernel<<<fuse_blocks, fuse_threads>>>(
        output.data_ptr<float>(),
        output.data_ptr<float>(), // In-place operation
        conv_numel,
        channel_size
    );
}
"""

# C++ interface
cpp_source = r"""
#include <torch/extension.h>

void fused_conv_transpose3d_softmax_sigmoid_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int stride,
    int padding,
    int output_padding,
    int dilation,
    int softmax_dim
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_conv_transpose3d_softmax_sigmoid_forward, 
          "Fused conv_transpose3d + softmax + sigmoid operation");
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
    softmax_dim,
):
    # Only support groups=1 for this implementation
    if conv_transpose_groups != 1:
        raise ValueError("Only conv_transpose_groups=1 is supported")
    
    # Calculate output dimensions
    batch_size, in_channels, D, H, W = x.shape
    out_channels = conv_transpose_weight.shape[0]
    kernel_size = conv_transpose_weight.shape[2]
    
    # Compute output dimensions
    out_d = (D - 1) * conv_transpose_stride + conv_transpose_output_padding - 2 * conv_transpose_padding + kernel_size
    out_h = (H - 1) * conv_transpose_stride + conv_transpose_output_padding - 2 * conv_transpose_padding + kernel_size
    out_w = (W - 1) * conv_transpose_stride + conv_transpose_output_padding - 2 * conv_transpose_padding + kernel_size
    
    # Create output tensor
    output = torch.empty((batch_size, out_channels, out_d, out_h, out_w), device=x.device, dtype=x.dtype)
    
    # Apply fused operations
    fused_ext.fused_op(
        x, 
        conv_transpose_weight, 
        conv_transpose_bias, 
        output,
        conv_transpose_stride,
        conv_transpose_padding,
        conv_transpose_output_padding,
        conv_transpose_dilation,
        softmax_dim
    )
    
    return output


batch_size = 16
in_channels = 32
out_channels = 64
D, H, W = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding]

def get_inputs():
    return [torch.rand(batch_size, in_channels, D, H, W)]
