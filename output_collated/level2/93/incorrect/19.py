# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_153448/code_4.py
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
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# The CUDA source includes a fused transposed convolution with add, min, GELU, and multiply operations.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__device__ __forceinline__ float fast_gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

__global__ void fused_conv_transpose_gelu_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int in_height,
    int in_width,
    int out_channels,
    int out_height,
    int out_width,
    float add_val,
    float mul_val
) {
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_ch = blockIdx.z;

    if (out_x >= out_width || out_y >= out_height || out_ch >= out_channels) return;

    float acc = 0.0f;
    int kernel_size = 4;
    int stride = 2;

    // Calculate corresponding input position
    int start_x = (out_x < kernel_size - 1) ? 0 : (out_x - kernel_size + 1 + stride - 1) / stride;
    int end_x = min((out_x + stride - 1) / stride, in_width);
    int start_y = (out_y < kernel_size - 1) ? 0 : (out_y - kernel_size + 1 + stride - 1) / stride;
    int end_y = min((out_y + stride - 1) / stride, in_height);

    for (int in_ch = 0; in_ch < in_channels; ++in_ch) {
        for (int k_y = start_y; k_y < end_y; ++k_y) {
            for (int k_x = start_x; k_x < end_x; ++k_x) {
                int k_idx_x = out_x - k_x * stride;
                int k_idx_y = out_y - k_y * stride;
                
                if (k_idx_x >= 0 && k_idx_x < kernel_size && k_idx_y >= 0 && k_idx_y < kernel_size) {
                    int input_idx = ((0 * in_channels + in_ch) * in_height + k_y) * in_width + k_x;
                    int weight_idx = ((out_ch * in_channels + in_ch) * kernel_size + k_idx_y) * kernel_size + k_idx_x;
                    acc += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    acc += bias[out_ch];
    
    // Apply fused operations: add -> min -> gelu -> multiply
    float val = acc + add_val;
    val = fminf(val, 0.0f);
    val = fast_gelu(val);
    val = val * mul_val;

    int output_idx = ((0 * out_channels + out_ch) * out_height + out_y) * out_width + out_x;
    output[output_idx] = val;
}

void fused_op_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    float add_val,
    float mul_val
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_height = input.size(2);
    int in_width = input.size(3);
    
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);
    int stride = 2;
    
    int out_height = (in_height - 1) * stride + kernel_size - 2 * 1 /*padding*/ + 1 /*output_padding*/;
    int out_width = (in_width - 1) * stride + kernel_size - 2 * 1 /*padding*/ + 1 /*output_padding*/;
    
    dim3 block(16, 16, 1);
    dim3 grid(
        (out_width + block.x - 1) / block.x,
        (out_height + block.y - 1) / block.y,
        out_channels
    );
    
    fused_conv_transpose_gelu_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        in_height,
        in_width,
        out_channels,
        out_height,
        out_width,
        add_val,
        mul_val
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    float add_val,
    float mul_val
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused conv transpose + add + min + GELU + multiply operation");
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
    add_value,
    multiply_value,
):
    # Ensure inputs are on CUDA
    x = x.contiguous()
    conv_transpose_weight = conv_transpose_weight.contiguous()
    conv_transpose_bias = conv_transpose_bias.contiguous()
    
    # Calculate output dimensions
    batch_size, in_channels, in_height, in_width = x.shape
    out_channels, _, kernel_size, _ = conv_transpose_weight.shape
    
    # Compute output dimensions matching F.conv_transpose2d behavior
    out_height = (in_height - 1) * conv_transpose_stride[0] - 2 * conv_transpose_padding[0] + kernel_size + conv_transpose_output_padding[0]
    out_width = (in_width - 1) * conv_transpose_stride[1] - 2 * conv_transpose_padding[1] + kernel_size + conv_transpose_output_padding[1]
    
    # Create output tensor
    out = torch.empty((batch_size, out_channels, out_height, out_width), device=x.device, dtype=x.dtype)
    
    # Launch fused kernel
    fused_ext.fused_op(x, conv_transpose_weight, conv_transpose_bias, out, float(add_value), float(multiply_value))
    
    return out

# Constants provided by original context
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
    return [torch.rand(batch_size, in_channels, height, width, device='cuda')]
