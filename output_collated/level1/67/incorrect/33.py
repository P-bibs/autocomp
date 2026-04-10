# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_161952/code_6.py
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

# ------------------------------------------------------------------
# CUDA kernel: Fused Conv1d + Bias
# ------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv1d_fused_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int N, const int C_in, const int C_out,
    const int L_in, const int L_out, const int K,
    const int stride, const int padding, const int dilation)
{
    // Global thread index
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = N * C_out * L_out;
    
    if (tid >= total_elements) return;

    // Mapping flat ID to (Batch, OutChannel, OutLength)
    const int n = tid / (C_out * L_out);
    const int n_rem = tid % (C_out * L_out);
    const int c_out = n_rem / L_out;
    const int l_out = n_rem % L_out;

    float acc = 0.0f;
    
    // Perform convolution
    // input layout: [N, C_in, L_in] 
    // weight layout: [C_out, C_in, K]
    for (int c_in = 0; c_in < C_in; ++c_in) {
        for (int k = 0; k < K; ++k) {
            int in_idx = l_out * stride + k * dilation - padding;
            if (in_idx >= 0 && in_idx < L_in) {
                float val = input[(n * C_in + c_in) * L_in + in_idx];
                float w = weight[(c_out * C_in + c_in) * K + k];
                acc += val * w;
            }
        }
    }

    // Add bias
    acc += bias[c_out];

    // Write output
    output[tid] = acc;
}

void conv1d_fused_forward(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor output, int stride, int padding, int dilation)
{
    const int N = input.size(0);
    const int C_out = weight.size(0);
    const int L_out = output.size(2);
    const int total_elements = N * C_out * L_out;

    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;

    conv1d_fused_forward_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, input.size(1), C_out,
        input.size(2), L_out, weight.size(2),
        stride, padding, dilation
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>

void conv1d_fused_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                          torch::Tensor output, int stride, int padding, int dilation);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv1d_fused_forward", &conv1d_fused_forward, "Fused 1D Conv and Bias");
}
"""

# Compile the extension
fused_ext = load_inline(
    name='fused_conv1d_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
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
    assert conv1d_groups == 1, "Only groups=1 supported"
    N, C_in, L_in = x.shape
    C_out, _, K = conv1d_weight.shape
    
    L_out = (L_in + 2 * conv1d_padding - conv1d_dilation * (K - 1) - 1) // conv1d_stride + 1
    out = torch.empty((N, C_out, L_out), device=x.device, dtype=x.dtype)
    
    fused_ext.conv1d_fused_forward(
        x, conv1d_weight, conv1d_bias, out, 
        conv1d_stride, conv1d_padding, conv1d_dilation
    )
    return out

# Helper functions provided by original context
batch_size = 32
in_channels = 64
out_channels = 128
kernel_size = 3
length = 131072

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]

def get_inputs():
    x = torch.rand(batch_size, in_channels, length, device='cuda', dtype=torch.float32)
    return [x]
