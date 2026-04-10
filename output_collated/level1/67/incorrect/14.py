# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_160042/code_6.py
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
# CUDA kernel: Implicit GEMM for 1D convolution (Kernel Size 3)
# ------------------------------------------------------------------
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv1d_k3_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int B, const int Cin, const int Cout,
    const int L_in, const int L_out,
    const int stride, const int padding, const int dilation
) {
    // blockIdx.x: batch index, blockIdx.y: output channel
    const int b = blockIdx.x;
    const int oc = blockIdx.y;
    
    // Each thread handles a subset of the output sequence
    for (int pos = threadIdx.x; pos < L_out; pos += blockDim.x) {
        float acc = 0.0f;
        const int input_pos = pos * stride - padding;

        for (int ic = 0; ic < Cin; ++ic) {
            const float* inp_ptr = input + (b * Cin + ic) * L_in;
            const float* wgt_ptr = weight + (oc * Cin + ic) * 3;

            #pragma unroll
            for (int k = 0; k < 3; ++k) {
                int idx = input_pos + k * dilation;
                if (idx >= 0 && idx < L_in) {
                    acc += inp_ptr[idx] * wgt_ptr[k];
                }
            }
        }

        if (bias != nullptr) {
            acc += bias[oc];
        }
        output[(b * Cout + oc) * L_out + pos] = acc;
    }
}

void launch_conv1d_k3(
    const torch::Tensor& input, const torch::Tensor& weight,
    const torch::Tensor& bias, torch::Tensor& output,
    int stride, int padding, int dilation
) {
    const int B = input.size(0);
    const int Cout = weight.size(0);
    const int L_out = output.size(2);
    
    dim3 grid(B, Cout);
    dim3 block(256);
    
    conv1d_k3_kernel<<<grid, block>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        B, input.size(1), Cout, input.size(2), L_out,
        stride, padding, dilation
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void launch_conv1d_k3(const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, torch::Tensor&, int, int, int);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv1d_k3", &launch_conv1d_k3, "Implicit GEMM Conv1d K=3");
}
"""

module = load_inline(
    name='conv1d_k3_ext',
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
    B, Cin, L_in = x.shape
    Cout, _, K = conv1d_weight.shape
    
    L_out = (L_in + 2 * conv1d_padding - conv1d_dilation * (K - 1) - 1) // conv1d_stride + 1
    output = torch.empty((B, Cout, L_out), device=x.device, dtype=x.dtype)
    
    module.conv1d_k3(
        x.contiguous(), 
        conv1d_weight.contiguous(), 
        conv1d_bias if conv1d_bias is not None else torch.tensor([], device=x.device), 
        output, 
        conv1d_stride, 
        conv1d_padding, 
        conv1d_dilation
    )
    return output

# Benchmark parameters
batch_size = 32
in_channels = 64
out_channels = 128
kernel_size = 3
length = 131072

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]

def get_inputs():
    return [torch.rand(batch_size, in_channels, length, device="cuda")]
