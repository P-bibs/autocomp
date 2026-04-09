# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_094451/code_0.py
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

# CUDA kernel for fused conv transpose + softmax + sigmoid
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_conv_transpose_softmax_sigmoid_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int D, int H, int W,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    int groups,
    int dilation,
    int softmax_dim
) {
    // Calculate output dimensions
    int out_D = (D - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1;
    int out_H = (H - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1;
    int out_W = (W - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1;
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * out_D * out_H * out_W;
    
    if (tid >= total_elements) return;
    
    // Decode output indices
    int temp = tid;
    int w_idx = temp % out_W; temp /= out_W;
    int h_idx = temp % out_H; temp /= out_H;
    int d_idx = temp % out_D; temp /= out_D;
    int c_idx = temp % out_channels; temp /= out_channels;
    int b_idx = temp;
    
    // Conv transpose calculation
    float sum = (bias != nullptr) ? bias[c_idx] : 0.0f;
    
    // Calculate input ranges
    int kD_min = max(0, (d_idx + padding - dilation * (kernel_size - 1) + stride - 1) / stride);
    int kD_max = min(D - 1, (d_idx + padding) / stride);
    
    int kH_min = max(0, (h_idx + padding - dilation * (kernel_size - 1) + stride - 1) / stride);
    int kH_max = min(H - 1, (h_idx + padding) / stride);
    
    int kW_min = max(0, (w_idx + padding - dilation * (kernel_size - 1) + stride - 1) / stride);
    int kW_max = min(W - 1, (w_idx + padding) / stride);
    
    for (int kd = kD_min; kd <= kD_max; kd++) {
        for (int kh = kH_min; kh <= kH_max; kh++) {
            for (int kw = kW_min; kw <= kW_max; kw++) {
                int k_idx_d = d_idx + padding - kd * stride;
                int k_idx_h = h_idx + padding - kh * stride;
                int k_idx_w = w_idx + padding - kw * stride;
                
                if (k_idx_d % dilation == 0 && k_idx_h % dilation == 0 && k_idx_w % dilation == 0) {
                    k_idx_d /= dilation;
                    k_idx_h /= dilation;
                    k_idx_w /= dilation;
                    
                    if (k_idx_d >= 0 && k_idx_d < kernel_size &&
                        k_idx_h >= 0 && k_idx_h < kernel_size &&
                        k_idx_w >= 0 && k_idx_w < kernel_size) {
                        
                        int g_idx = c_idx / (out_channels / groups);
                        int input_idx = ((((b_idx * in_channels) + (g_idx * (in_channels / groups) + (c_idx % (in_channels / groups)))) * D + kd) * H + kh) * W + kw;
                        int weight_idx = (((((g_idx * (out_channels / groups)) + (c_idx % (out_channels / groups))) * in_channels / groups) + (c_idx % (in_channels / groups))) * kernel_size + k_idx_d) * kernel_size * kernel_size + k_idx_h * kernel_size + k_idx_w;
                        
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }
    
    // Apply sigmoid
    float sigmoid_val = 1.0f / (1.0f + expf(-sum));
    output[tid] = sigmoid_val;
}

void fused_conv_transpose_softmax_sigmoid_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    int groups,
    int dilation,
    int softmax_dim
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int D = input.size(2);
    int H = input.size(3);
    int W = input.size(4);
    
    int out_D = (D - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1;
    int out_H = (H - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1;
    int out_W = (W - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1;
    
    int total_elements = batch_size * weight.size(0) * out_D * out_H * out_W;
    
    int threads_per_block = 256;
    int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    fused_conv_transpose_softmax_sigmoid_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        weight.size(0),
        D, H, W,
        kernel_size,
        stride,
        padding,
        output_padding,
        groups,
        dilation,
        softmax_dim
    );
}
"""

# C++ interface
cpp_source = r"""
#include <torch/extension.h>

void fused_conv_transpose_softmax_sigmoid_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    int groups,
    int dilation,
    int softmax_dim
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_conv_transpose_softmax_sigmoid_forward, "Fused conv transpose, softmax, and sigmoid");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_conv_transpose_softmax_sigmoid',
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
    # Calculate output dimensions
    D, H, W = x.shape[2], x.shape[3], x.shape[4]
    kernel_size = conv_transpose_weight.shape[2]  # assuming cubic kernel
    
    out_D = (D - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_dilation * (kernel_size - 1) + conv_transpose_output_padding + 1
    out_H = (H - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_dilation * (kernel_size - 1) + conv_transpose_output_padding + 1
    out_W = (W - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_dilation * (kernel_size - 1) + conv_transpose_output_padding + 1
    
    out_channels = conv_transpose_weight.shape[0]
    batch_size = x.shape[0]
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, out_D, out_H, out_W, device=x.device, dtype=x.dtype)
    
    # Call fused kernel
    fused_ext.fused_op(
        x, 
        conv_transpose_weight, 
        conv_transpose_bias, 
        output,
        kernel_size,
        conv_transpose_stride,
        conv_transpose_padding,
        conv_transpose_output_padding,
        conv_transpose_groups,
        conv_transpose_dilation,
        softmax_dim
    )
    
    return output

# Constants
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
