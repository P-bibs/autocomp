# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_162105/code_3.py
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

# ----------------------------------------------------------------------
# CUDA source – fused transposed convolution and pointwise operations
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void fused_conv_transpose2d_pointwise_kernel(
    const float* __restrict__ input,
    const float* __restrict__ conv_weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const float scaling_factor,
    const int N, const int C_in, const int H_in, const int W_in,
    const int C_out, const int H_out, const int W_out,
    const int kernel_size, const int stride, const int padding, 
    const int output_padding, const int dilation) {
    
    // Calculate output indices
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C_out * H_out * W_out;
    if (idx >= total_elements) return;
    
    // Decompose flat index to n, c_out, h_out, w_out
    int n = idx / (C_out * H_out * W_out);
    int temp = idx % (C_out * H_out * W_out);
    int c_out = temp / (H_out * W_out);
    temp = temp % (H_out * W_out);
    int h_out = temp / W_out;
    int w_out = temp % W_out;
    
    // Perform transposed convolution calculation
    float conv_result = 0.0f;
    
    // Loop over input channels
    for (int c_in = 0; c_in < C_in; c_in++) {
        // Loop over kernel elements
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                // Calculate corresponding input position
                int h_in = h_out - stride * kh + 2 * padding;
                int w_in = w_out - stride * kw + 2 * padding;
                
                // Check if input position is valid
                if (h_in % stride == 0 && w_in % stride == 0) {
                    h_in /= stride;
                    w_in /= stride;
                    
                    if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                        int input_idx = ((n * C_in + c_in) * H_in + h_in) * W_in + w_in;
                        int weight_idx = ((c_in * C_out + c_out) * kernel_size + kh) * kernel_size + kw;
                        conv_result += input[input_idx] * conv_weight[weight_idx];
                    }
                }
            }
        }
    }
    
    // Add convolution bias
    conv_result += conv_bias[c_out];
    
    // Apply pointwise operations
    float v = conv_result;
    
    // 1) Add per-channel bias
    v += bias[c_out];
    
    // 2) First clamp to [0, 1]
    if (v < 0.0f) v = 0.0f;
    else if (v > 1.0f) v = 1.0f;
    
    // 3) Scale
    v = v * scaling_factor;
    
    // 4) Second clamp to [0, 1]
    if (v < 0.0f) v = 0.0f;
    else if (v > 1.0f) v = 1.0f;
    
    // 5) Un-scale
    v = v / scaling_factor;
    
    // Write result
    output[idx] = v;
}

// Host function that launches the kernel
void fused_conv_transpose2d_pointwise_forward(
    torch::Tensor input,
    torch::Tensor conv_weight,
    torch::Tensor conv_bias,
    torch::Tensor bias,
    torch::Tensor output,
    float scaling_factor,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    int dilation) {
    
    const int N = input.size(0);
    const int C_in = input.size(1);
    const int H_in = input.size(2);
    const int W_in = input.size(3);
    
    const int C_out = output.size(1);
    const int H_out = output.size(2);
    const int W_out = output.size(3);
    
    const int total_elements = N * C_out * H_out * W_out;
    const int block_size = 256;
    const int grid_size = (total_elements + block_size - 1) / block_size;
    
    fused_conv_transpose2d_pointwise_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        conv_weight.data_ptr<float>(),
        conv_bias.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        scaling_factor,
        N, C_in, H_in, W_in,
        C_out, H_out, W_out,
        kernel_size, stride, padding, output_padding, dilation);
}
"""

# ----------------------------------------------------------------------
# C++ source – pybind11 binding
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void fused_conv_transpose2d_pointwise_forward(
    torch::Tensor input,
    torch::Tensor conv_weight,
    torch::Tensor conv_bias,
    torch::Tensor bias,
    torch::Tensor output,
    float scaling_factor,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    int dilation);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_transpose2d_pointwise", &fused_conv_transpose2d_pointwise_forward, 
          "Fused transposed convolution and pointwise operations");
}
"""

# ----------------------------------------------------------------------
# Compile the inline CUDA extension
# ----------------------------------------------------------------------
fused_ext = load_inline(
    name='fused_conv_transpose2d_pointwise',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True,
)

# ----------------------------------------------------------------------
# Optimised functional_model
# ----------------------------------------------------------------------
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
    # Calculate output dimensions
    N, C_in, H_in, W_in = x.shape
    C_out = conv_transpose_weight.shape[1]  # Note: for conv_transpose, weight shape is (in_channels, out_channels/groups, kH, kW)
    
    # For transposed convolution output size calculation:
    H_out = (H_in - 1) * conv_transpose_stride[0] - 2 * conv_transpose_padding[0] + conv_transpose_weight.shape[2] + conv_transpose_output_padding[0]
    W_out = (W_in - 1) * conv_transpose_stride[1] - 2 * conv_transpose_padding[1] + conv_transpose_weight.shape[3] + conv_transpose_output_padding[1]
    
    # Create output tensor
    output = torch.empty(N, C_out, H_out, W_out, device=x.device, dtype=x.dtype)
    
    # Call fused kernel
    fused_ext.fused_conv_transpose2d_pointwise(
        x,
        conv_transpose_weight,
        conv_transpose_bias,
        bias,
        output,
        scaling_factor,
        conv_transpose_weight.shape[2],  # kernel_size (assuming square kernel)
        conv_transpose_stride[0],
        conv_transpose_padding[0],
        conv_transpose_output_padding[0],
        conv_transpose_dilation[0]
    )
    
    return output

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
