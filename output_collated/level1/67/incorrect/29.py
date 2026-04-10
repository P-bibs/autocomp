# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_161952/code_3.py
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
# Inline CUDA kernel – 1‑D convolution (batch, in_channels, length) → (batch, out_channels, length)
# -------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv1d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch,
    const int in_channels,
    const int out_channels,
    const int in_length,
    const int out_length,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation,
    const int groups)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch * out_channels * out_length) return;

    // unravel flat index → (batch, out_channel, out_position)
    int out_pos   = idx % out_length;
    int tmp       = idx / out_length;
    int out_ch    = tmp % out_channels;
    int batch_idx = tmp / out_channels;

    // grouped convolution parameters
    int out_ch_per_group = out_channels / groups;
    int in_ch_per_group  = in_channels / groups;
    int grp_id = out_ch / out_ch_per_group;
    int in_ch_offset = grp_id * in_ch_per_group;

    // start with bias (bias is always non‑null – a zero tensor is supplied when the user does not provide one)
    float sum = bias[out_ch];

    // loop over kernel elements
    for (int k = 0; k < kernel_size; ++k) {
        long long input_index = (long long)out_pos * stride + (long long)k * dilation - padding;
        if (input_index >= 0 && input_index < in_length) {
            // accumulate over all input channels that belong to the same group
            for (int ic = 0; ic < in_ch_per_group; ++ic) {
                int in_ch = in_ch_offset + ic;
                // input layout: (batch, in_channels, length)
                int inp_offset = ((batch_idx * in_channels + in_ch) * in_length + (int)input_index);
                float v_in = input[inp_offset];
                // weight layout: (out_channels, in_ch_per_group, kernel_size)
                int w_offset = ((out_ch * in_ch_per_group + ic) * kernel_size + k);
                float v_w = weight[w_offset];
                sum += v_in * v_w;
            }
        }
    }

    // write result – output layout: (batch, out_channels, out_length)
    int out_offset = ((batch_idx * out_channels + out_ch) * out_length + out_pos);
    output[out_offset] = sum;
}

/* Host (Python‑callable) wrapper */
void conv1d_forward(
    at::Tensor input,
    at::Tensor weight,
    at::Tensor bias,
    at::Tensor output,
    int stride,
    int padding,
    int dilation,
    int groups)
{
    const int block_size = 256;
    int N = output.size(0) * output.size(1) * output.size(2);
    int grid = (N + block_size - 1) / block_size;

    conv1d_kernel<<<grid, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        input.size(0),               // batch
        input.size(1),               // in_channels
        weight.size(0),              // out_channels
        input.size(2),               // in_length
        output.size(2),              // out_length
        weight.size(2),              // kernel_size
        stride,
        padding,
        dilation,
        groups);
    cudaDeviceSynchronize();
}
"""

# -------------------------------------------------------------------------
# C++ bindings – expose the CUDA function to Python
# -------------------------------------------------------------------------
cpp_source = r"""
#include <torch/extension.h>

void conv1d_forward(
    at::Tensor input,
    at::Tensor weight,
    at::Tensor bias,
    at::Tensor output,
    int stride,
    int padding,
    int dilation,
    int groups);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv1d_forward, "1‑D convolution forward");
}
"""

# Build the inline extension
conv_ext = load_inline(
    name='conv_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

# -------------------------------------------------------------------------
# Parameters used only for illustration in get_init_inputs / get_inputs
# -------------------------------------------------------------------------
batch_size = 32
in_channels = 64
out_channels = 128
kernel_size = 3
length = 131072

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]

def get_inputs():
    # Random input – will be moved to GPU inside functional_model
    x = torch.rand(batch_size, in_channels, length)
    return [x]

# -------------------------------------------------------------------------
# Functional model – replaces the original PyTorch conv1d with the custom CUDA kernel
# -------------------------------------------------------------------------
def functional_model(
    x,
    *,
    conv1d_weight,
    conv1d_bias,
    conv1d_stride,
    conv1d_padding,
    conv1d_dilation,
    conv1d_groups):
    # -----------------------------------------------------------------
    # 1. Move data to the GPU
    # -----------------------------------------------------------------
    x = x.cuda()
    conv1d_weight = conv1d_weight.cuda()
    # If the caller did not provide a bias, use a zero tensor (the kernel always adds it)
    if conv1d_bias is None:
        bias = torch.zeros(conv1d_weight.shape[0], device=x.device, dtype=x.dtype)
    else:
        bias = conv1d_bias.cuda()

    # -----------------------------------------------------------------
    # 2. Compute output length using the same formula as PyTorch
    # -----------------------------------------------------------------
    k = conv1d_weight.shape[2]
    L_in = x.shape[2]
    L_out = (L_in + 2 * conv1d_padding - conv1d_dilation * (k - 1) - 1) // conv1d_stride + 1

    # -----------------------------------------------------------------
    # 3. Allocate output tensor on the device
    # -----------------------------------------------------------------
    out = torch.zeros(x.shape[0], conv1d_weight.shape[0], L_out, device=x.device, dtype=x.dtype)

    # -----------------------------------------------------------------
    # 4. Launch the custom CUDA kernel
    # -----------------------------------------------------------------
    conv_ext.forward(
        x,
        conv1d_weight,
        bias,
        out,
        conv1d_stride,
        conv1d_padding,
        conv1d_dilation,
        conv1d_groups)

    return out
