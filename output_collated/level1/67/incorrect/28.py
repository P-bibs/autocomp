# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_161448/code_6.py
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

# ----------------------------------------------------------------------
# 1. CUDA kernel that implements 1-D convolution (global-memory only)
# ----------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv1d_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int B, const int Cin, const int Cout,
    const int K, const int L, const int Lout,
    const int stride, const int padding,
    const int dilation, const int groups)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * Cout * Lout) return;

    int tmp = idx;
    const int out_pos = tmp % Lout; tmp /= Lout;
    const int oc      = tmp % Cout; tmp /= Cout;
    const int b       = tmp;

    const int groups_per_oc = Cout / groups;
    const int group_id      = oc / groups_per_oc;
    const int ic_per_group  = Cin / groups;
    const int weight_group_offset = oc * (ic_per_group * K);
    const int input_group_offset = group_id * ic_per_group * L;

    float sum = bias ? bias[oc] : 0.0f;

    for (int ic = 0; ic < ic_per_group; ++ic) {
        for (int k = 0; k < K; ++k) {
            const int in_pos = out_pos * stride - padding + k * dilation;
            if (in_pos >= 0 && in_pos < L) {
                float val = input[b * (Cin * L) + (group_id * ic_per_group + ic) * L + in_pos];
                float w   = weight[weight_group_offset + ic * K + k];
                sum += val * w;
            }
        }
    }
    output[idx] = sum;
}

void conv1d_forward_launch(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output,
    int stride, int padding, int dilation, int groups)
{
    const int B = input.size(0);
    const int Cin = input.size(1);
    const int Cout = weight.size(0);
    const int L = input.size(2);
    const int K = weight.size(2);
    const int Lout = output.size(2);

    const int total = B * Cout * Lout;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    conv1d_forward_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), 
        bias.defined() ? bias.data_ptr<float>() : nullptr, output.data_ptr<float>(),
        B, Cin, Cout, K, L, Lout, stride, padding, dilation, groups
    );
}
"""

# ----------------------------------------------------------------------
# 2. C++ binding
# ----------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void conv1d_forward_launch(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int, int, int, int);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv1d_forward", &conv1d_forward_launch, "Custom 1D Conv CUDA");
}
"""

# Compile
fused_ext = load_inline(
    name='custom_conv1d',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(
    x, *, conv1d_weight, conv1d_bias, conv1d_stride,
    conv1d_padding, conv1d_dilation, conv1d_groups,
):
    L = x.shape[2]
    K = conv1d_weight.shape[2]
    Lout = (L + 2 * conv1d_padding - conv1d_dilation * (K - 1) - 1) // conv1d_stride + 1
    
    out = torch.empty((x.shape[0], conv1d_weight.shape[0], Lout), device=x.device, dtype=x.dtype)
    
    fused_ext.conv1d_forward(
        x, conv1d_weight, conv1d_bias, out,
        conv1d_stride, conv1d_padding, conv1d_dilation, conv1d_groups
    )
    return out

# Placeholders for original script compatibility
batch_size = 32
in_channels = 64
out_channels = 128
kernel_size = 3
length = 131072

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]

def get_inputs():
    return [torch.rand(batch_size, in_channels, length, device='cuda', dtype=torch.float32)]
