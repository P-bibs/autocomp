# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_122448/code_6.py
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

# Optimization: Fused CUDA kernel for clamp and division
# Note: In a production environment, one would implement a custom 3D transposed convolution kernel.
# However, given the constraint of providing a single high-performance implementation,
# we focus on the fused post-processing kernel while leveraging ATen's highly-optimized
# low-level convolution engine for the compute-heavy phase.

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <algorithm>

__global__ void fused_clamp_div_kernel(float* data, float min_val, float divisor, int numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        float val = data[idx];
        val = (val < min_val) ? min_val : val;
        data[idx] = val / divisor;
    }
}

void fused_op_cuda(torch::Tensor x, float min_val, float divisor) {
    int numel = x.numel();
    int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    fused_clamp_div_kernel<<<blocks, threads>>>(x.data_ptr<float>(), min_val, divisor, numel);
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_op_cuda(torch::Tensor x, float min_val, float divisor);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_cuda, "Fused clamp and division kernel");
}
"""

# Compile the fused kernel extension
fused_ext = load_inline(
    name='fused_ext',
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
    # Perform convolution as the compute-heavy primitive
    x = torch.nn.functional.conv_transpose3d(
        x, 
        conv_transpose_weight, 
        conv_transpose_bias, 
        stride=conv_transpose_stride, 
        padding=conv_transpose_padding, 
        output_padding=conv_transpose_output_padding, 
        groups=conv_transpose_groups, 
        dilation=conv_transpose_dilation
    )
    
    # Ensure tensor is contiguous for the CUDA kernel
    if not x.is_contiguous():
        x = x.contiguous()
        
    # Execute fused kernel to shift from O(N) launches to O(1)
    fused_ext.fused_op(x, float(min_value), float(divisor))
    
    return x

# Configuration for evaluation
batch_size, in_channels, out_channels = 16, 64, 128
depth, height, width = 24, 48, 48
kernel_size, stride, padding = 3, 2, 1
min_value, divisor = -1.0, 2.0

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, min_value, divisor]

def get_inputs():
    # Return device-ready tensors
    return [torch.rand(batch_size, in_channels, depth, height, width, device='cuda')]
