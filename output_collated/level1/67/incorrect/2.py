# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_155306/code_0.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'dilation', 'groups', 'bias']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv1d_weight', 'conv1d_bias', 'conv1d_stride', 'conv1d_padding', 'conv1d_dilation', 'conv1d_groups']
REQUIRED_FLAT_STATE_NAMES = ['conv1d_weight', 'conv1d_bias']


class ModelNew(nn.Module):
    """
    Performs a standard 1D convolution operation.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int=1, padding: int=0, dilation: int=1, groups: int=1, bias: bool=False):
        super(ModelNew, self).__init__()
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

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
    # State for conv1d (nn.Conv1d)
    if 'conv1d_weight' in flat_state:
        state_kwargs['conv1d_weight'] = flat_state['conv1d_weight']
    else:
        state_kwargs['conv1d_weight'] = getattr(model.conv1d, 'weight', None)
    if 'conv1d_bias' in flat_state:
        state_kwargs['conv1d_bias'] = flat_state['conv1d_bias']
    else:
        state_kwargs['conv1d_bias'] = getattr(model.conv1d, 'bias', None)
    state_kwargs['conv1d_stride'] = model.conv1d.stride
    state_kwargs['conv1d_padding'] = model.conv1d.padding
    state_kwargs['conv1d_dilation'] = model.conv1d.dilation
    state_kwargs['conv1d_groups'] = model.conv1d.groups
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

# CUDA kernel implementing Conv1d manually
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv1d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_length,
    int kernel_size,
    int output_length,
    int stride,
    int padding,
    int dilation
) {
    // Each thread handles one output element
    int batch_idx = blockIdx.x;
    int out_ch = blockIdx.y;
    int out_pos = blockIdx.z * blockDim.x + threadIdx.x;

    if (batch_idx >= batch_size || out_ch >= out_channels || out_pos >= output_length)
        return;

    float sum = 0.0f;
    for (int k = 0; k < kernel_size; ++k) {
        int input_pos = out_pos * stride - padding + k * dilation;
        if (input_pos >= 0 && input_pos < input_length) {
            for (int in_ch = 0; in_ch < in_channels; ++in_ch) {
                sum += input[batch_idx * in_channels * input_length + in_ch * input_length + input_pos] *
                       weight[out_ch * in_channels * kernel_size + in_ch * kernel_size + k];
            }
        }
    }

    // Add bias
    if (bias != nullptr) {
        sum += bias[out_ch];
    }

    output[batch_idx * out_channels * output_length + out_ch * output_length + out_pos] = sum;
}

void launch_conv1d_kernel(
    const torch::Tensor &input,
    const torch::Tensor &weight,
    const c10::optional<torch::Tensor> &bias,
    torch::Tensor &output,
    int stride,
    int padding,
    int dilation
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_length = input.size(2);
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);
    int output_length = output.size(2);

    dim3 grid(batch_size, out_channels, (output_length + 255) / 256);
    dim3 block(256);

    const float* input_ptr = input.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = bias.has_value() ? bias.value().data_ptr<float>() : nullptr;
    float* output_ptr = output.data_ptr<float>();

    conv1d_kernel<<<grid, block>>>(
        input_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        batch_size,
        in_channels,
        out_channels,
        input_length,
        kernel_size,
        output_length,
        stride,
        padding,
        dilation
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void launch_conv1d_kernel(
    const torch::Tensor &input,
    const torch::Tensor &weight,
    const c10::optional<torch::Tensor> &bias,
    torch::Tensor &output,
    int stride,
    int padding,
    int dilation
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv1d_kernel", &launch_conv1d_kernel, "Custom Conv1D kernel");
}
"""

# Compile the extension
conv1d_ext = load_inline(
    name='conv1d_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# Reimplement functional_model using our custom CUDA kernel
def functional_model(
    x,
    *,
    conv1d_weight,
    conv1d_bias,
    conv1d_stride,
    conv1d_padding,
    conv1d_dilation,
    conv1d_groups,
):
    # Support for groups is not implemented yet, so we keep the assertion
    assert conv1d_groups == 1, "Groups not supported in this implementation"

    batch_size, in_channels, length = x.shape
    out_channels, _, kernel_size = conv1d_weight.shape
    
    # Calculate output length based on the formula for conv1d
    output_length = ((length + 2 * conv1d_padding - conv1d_dilation * (kernel_size - 1) - 1) // conv1d_stride) + 1
    output = torch.empty((batch_size, out_channels, output_length), device=x.device, dtype=x.dtype)

    conv1d_ext.conv1d_kernel(x, conv1d_weight, conv1d_bias, output, conv1d_stride, conv1d_padding, conv1d_dilation)
    return output


# Test data parameters
batch_size = 32
in_channels = 64
out_channels = 128
kernel_size = 3
length = 131072

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]

def get_inputs():
    x = torch.rand(batch_size, in_channels, length, device='cuda')
    return [x]

# Move model parameters to GPU for use in functional_model
def setup_model_params():
    global conv1d_weight, conv1d_bias
    conv1d_weight = torch.rand(out_channels, in_channels, kernel_size, device='cuda')
    conv1d_bias = torch.rand(out_channels, device='cuda')
    return {
        'conv1d_weight': conv1d_weight,
        'conv1d_bias': conv1d_bias,
        'conv1d_stride': 1,
        'conv1d_padding': 0,
        'conv1d_dilation': 1,
        'conv1d_groups': 1
    }
