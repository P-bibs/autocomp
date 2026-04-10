# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_154319/code_3.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'add_value', 'multiply_value']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'add_value', 'multiply_value']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a transposed convolution, adds a value, takes the minimum, applies GELU, and multiplies by a value.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, add_value, multiply_value):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.add_value = add_value
        self.multiply_value = multiply_value

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
    if 'add_value' in flat_state:
        state_kwargs['add_value'] = flat_state['add_value']
    else:
        state_kwargs['add_value'] = getattr(model, 'add_value')
    if 'multiply_value' in flat_state:
        state_kwargs['multiply_value'] = flat_state['multiply_value']
    else:
        state_kwargs['multiply_value'] = getattr(model, 'multiply_value')
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

# CUDA kernel implementing fused conv transpose + elementwise operations
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_op_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    float add_value,
    float multiply_value,
    int B, int Ci, int Co, int Hi, int Wi, int Hw, int Ww
) {
    int Ho = (Hi - 1) * 2 + Hw - 2; // stride=2, padding=1
    int Wo = (Wi - 1) * 2 + Ww - 2;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * Co * Ho * Wo;
    
    if (idx >= total_elements) return;
    
    // Calculate output indices
    int tmp = idx;
    int wo = tmp % Wo; tmp /= Wo;
    int ho = tmp % Ho; tmp /= Ho;
    int co = tmp % Co;
    int b  = tmp / Co;
    
    float acc = 0.0f;
    
    // Conv transpose logic for stride=2, padding=1
    for (int ci = 0; ci < Ci; ++ci) {
        for (int kh = 0; kh < Hw; ++kh) {
            for (int kw = 0; kw < Ww; ++kw) {
                // Map output position to input position
                int hi = (ho + 1 - kh) / 2; // inverse of: ho = hi*2 + kh - 1
                int wi = (wo + 1 - kw) / 2;
                
                // Check bounds and stride condition
                if (hi >= 0 && hi < Hi && wi >= 0 && wi < Wi &&
                    (ho + 1 - kh) % 2 == 0 && (wo + 1 - kw) % 2 == 0) {
                    float x_val = input[((b * Ci + ci) * Hi + hi) * Wi + wi];
                    float w_val = weight[((ci * Co + co) * Hw + kh) * Ww + kw];
                    acc += x_val * w_val;
                }
            }
        }
    }
    
    // Add bias
    acc += bias[co];
    
    // Fused element-wise operations
    // 1. Add value
    acc += add_value;
    
    // 2. Min with 0
    acc = fminf(acc, 0.0f);
    
    // 3. GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    float cdf = 0.5f * (1.0f + tanhf(0.79788456f * (acc + 0.044715f * acc * acc * acc)));
    acc = acc * cdf;
    
    // 4. Multiply
    acc *= multiply_value;
    
    output[idx] = acc;
}

void fused_op_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    float add_value,
    float multiply_value
) {
    int B = input.size(0);
    int Ci = input.size(1);
    int Hi = input.size(2);
    int Wi = input.size(3);
    int Co = weight.size(0);
    int Hw = weight.size(2);
    int Ww = weight.size(3);
    
    int Ho = (Hi - 1) * 2 + Hw - 2; // stride=2, padding=1
    int Wo = (Wi - 1) * 2 + Ww - 2;
    
    int total_threads = B * Co * Ho * Wo;
    int block_size = 256;
    int grid_size = (total_threads + block_size - 1) / block_size;
    
    fused_op_forward_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        add_value,
        multiply_value,
        B, Ci, Co, Hi, Wi, Hw, Ww
    );
}
"""

# C++ binding
cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    float add_value,
    float multiply_value
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused ConvTranspose2d + Elementwise Ops");
}
"""

# Compile extension
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
    add_value,
    multiply_value,
):
    # Validate assumptions
    assert conv_transpose_stride == 2, "Only stride=2 is supported"
    assert conv_transpose_padding == 1, "Only padding=1 is supported"
    assert conv_transpose_groups == 1, "Only groups=1 is supported"
    assert conv_transpose_dilation == 1, "Only dilation=1 is supported"
    assert conv_transpose_output_padding == 0, "Only output_padding=0 is supported"
    
    # Calculate output dimensions
    B, Ci, Hi, Wi = x.shape
    Co, _, Hw, Ww = conv_transpose_weight.shape
    Ho = (Hi - 1) * conv_transpose_stride + Hw - 2 * conv_transpose_padding + conv_transpose_output_padding
    Wo = (Wi - 1) * conv_transpose_stride + Ww - 2 * conv_transpose_padding + conv_transpose_output_padding
    
    # Create output tensor
    output = torch.empty((B, Co, Ho, Wo), device=x.device, dtype=x.dtype)
    
    # Call fused kernel
    fused_ext.fused_op(x, conv_transpose_weight, conv_transpose_bias, output, add_value, multiply_value)
    
    return output

# Test parameters
batch_size = 128
in_channels = 64
out_channels = 128
height, width = 64, 64
kernel_size = 4
stride = 2
add_value = 0.5
multiply_value = 2.0

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, add_value, multiply_value]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]
