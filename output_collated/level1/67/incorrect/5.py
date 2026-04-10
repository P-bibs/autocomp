# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_155306/code_6.py
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

# Formula for output length of conv1d: floor((L + 2*pad - dil * (kernel - 1) - 1) / stride + 1)
def get_out_length(L, pad, dil, k, stride):
    return (L + 2 * pad - dil * (k - 1) - 1) // stride + 1

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv1d_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C_in, int C_out, int L_in, int L_out,
    int K, int stride, int padding, int dilation, int groups)
{
    int n = blockIdx.x;
    int cout = blockIdx.y;
    int tid = threadIdx.x;

    int C_in_per_group = C_in / groups;
    int g_idx = cout / (C_out / groups);
    int weight_offset = cout * C_in_per_group * K;

    for (int l_out = tid; l_out < L_out; l_out += blockDim.x) {
        float acc = (bias != nullptr) ? bias[cout] : 0.0f;
        int base = l_out * stride - padding;

        const float* w_ptr = weight + weight_offset;
        const float* i_base = input + n * C_in * L_in + g_idx * C_in_per_group * L_in;

        for (int c = 0; c < C_in_per_group; ++c) {
            const float* i_chan = i_base + c * L_in;
            const float* w_chan = w_ptr + c * K;
            for (int k = 0; k < K; ++k) {
                int idx = base + k * dilation;
                if (idx >= 0 && idx < L_in) {
                    acc += i_chan[idx] * w_chan[k];
                }
            }
        }
        output[n * C_out * L_out + cout * L_out + l_out] = acc;
    }
}

void conv1d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    int stride, int padding, int dilation, int groups)
{
    const int N = input.size(0);
    const int C_in = input.size(1);
    const int L_in = input.size(2);
    const int C_out = weight.size(0);
    const int K = weight.size(2);
    const int L_out = output.size(2);

    dim3 block(256);
    dim3 grid(N, C_out);

    conv1d_forward_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        N, C_in, C_out, L_in, L_out, K,
        stride, padding, dilation, groups);
}
"""

cpp_source = r"""
void conv1d_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                    torch::Tensor output, int stride, int padding, int dilation, int groups);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv1d_forward", &conv1d_forward, "Custom CUDA Conv1D");
}
"""

# Compile extension
conv1d_ext = load_inline(
    name="custom_conv1d",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    with_cuda=True,
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
    N, C_in, L_in = x.shape
    C_out, _, K = conv1d_weight.shape
    L_out = get_out_length(L_in, conv1d_padding, conv1d_dilation, K, conv1d_stride)
    
    output = torch.empty((N, C_out, L_out), device=x.device, dtype=x.dtype)
    
    conv1d_ext.conv1d_forward(
        x, conv1d_weight, conv1d_bias, output,
        conv1d_stride, conv1d_padding, conv1d_dilation, conv1d_groups
    )
    return output

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
