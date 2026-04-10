# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_161952/code_7.py
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

# -------------------------------------------------------------------------
# CUDA kernel: Optimized 1D Convolution with bias fusion
# -------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv1d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch, const int in_channels, const int out_channels,
    const int in_length, const int out_length,
    const int kernel_size, const int stride,
    const int padding, const int dilation, const int groups)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch * out_channels * out_length) return;

    int out_pos   = idx % out_length;
    int tmp       = idx / out_length;
    int out_ch    = tmp % out_channels;
    int batch_idx = tmp / out_channels;

    int in_ch_per_group = in_channels / groups;
    int grp_id = out_ch / (out_channels / groups);
    int in_ch_start = grp_id * in_ch_per_group;

    float sum = bias[out_ch];

    for (int k = 0; k < kernel_size; ++k) {
        int in_pos = out_pos * stride + k * dilation - padding;
        if (in_pos >= 0 && in_pos < in_length) {
            for (int ic = 0; ic < in_ch_per_group; ++ic) {
                int in_ch = in_ch_start + ic;
                float inp_val = input[((batch_idx * in_channels + in_ch) * in_length + in_pos)];
                float w_val = weight[((out_ch * in_ch_per_group + ic) * kernel_size + k)];
                sum += inp_val * w_val;
            }
        }
    }
    output[idx] = sum;
}

void conv1d_forward(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output,
    int stride, int padding, int dilation, int groups)
{
    const int out_elements = output.numel();
    const int threads = 256;
    const int blocks = (out_elements + threads - 1) / threads;

    conv1d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(),
        (int)input.size(0), (int)input.size(1), (int)weight.size(0),
        (int)input.size(2), (int)output.size(2),
        (int)weight.size(2), stride, padding, dilation, groups);
}
"""

cpp_source = r"""
void conv1d_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output,
                    int stride, int padding, int dilation, int groups);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv1d_forward, "1D Conv Forward");
}
"""

# Compile extension once
conv_ext = load_inline(
    name='conv_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

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
    # Ensure inputs are on GPU
    x = x.contiguous().cuda()
    weight = conv1d_weight.contiguous().cuda()
    bias = conv1d_bias.contiguous().cuda() if conv1d_bias is not None else torch.zeros(weight.shape[0], device='cuda')
    
    # Calculate output shape
    L_in = x.shape[2]
    k = weight.shape[2]
    L_out = (L_in + 2 * conv1d_padding - conv1d_dilation * (k - 1) - 1) // conv1d_stride + 1
    
    out = torch.empty((x.shape[0], weight.shape[0], L_out), device='cuda', dtype=x.dtype)
    
    # Launch custom kernel
    conv_ext.forward(x, weight, bias, out, conv1d_stride, conv1d_padding, conv1d_dilation, conv1d_groups)
    
    return out
