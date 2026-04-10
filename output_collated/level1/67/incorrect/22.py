# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_160800/code_4.py
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

# CUDA kernel for optimized 1D direct convolution
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void conv1d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    int batch_size, int in_channels, int out_channels,
    int input_length, int output_length,
    int kernel_size, int stride, int padding, int dilation
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size * out_channels * output_length) return;

    int out_idx = tid % output_length;
    int tmp = tid / output_length;
    int out_ch = tmp % out_channels;
    int b = tmp / out_channels;

    scalar_t acc = bias[out_ch];
    int in_start = out_idx * stride - padding;

    // Weight pointer for this channel
    const scalar_t* weight_ch = weight + out_ch * in_channels * kernel_size;

    for (int k = 0; k < kernel_size; ++k) {
        int in_pos = in_start + k * dilation;
        if (in_pos >= 0 && in_pos < input_length) {
            const scalar_t* input_ptr = input + (b * in_channels * input_length) + in_pos;
            const scalar_t* w_ptr = weight_ch + (k);
            
            for (int in_ch = 0; in_ch < in_channels; ++in_ch) {
                acc += input_ptr[in_ch * input_length] * w_ptr[in_ch * kernel_size];
            }
        }
    }
    output[tid] = acc;
}

void conv1d_forward(
    torch::Tensor x, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor output, int stride, int padding, int dilation
) {
    int batch_size = x.size(0);
    int in_channels = x.size(1);
    int input_length = x.size(2);
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);
    int output_length = output.size(2);

    int total_elements = batch_size * out_channels * output_length;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "conv1d_kernel", ([&] {
        conv1d_kernel<scalar_t><<<blocks, threads>>>(
            x.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
            batch_size, in_channels, out_channels,
            input_length, output_length, kernel_size,
            stride, padding, dilation
        );
    }));
}
"""

cpp_source = r"""
void conv1d_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias,
                    torch::Tensor output, int stride, int padding, int dilation);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv1d_forward", &conv1d_forward, "Optimized 1D Conv");
}
"""

conv_ext = load_inline(
    name='conv1d_opt', cpp_sources=cpp_source, cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True
)

def functional_model(x, *, conv1d_weight, conv1d_bias, conv1d_stride, 
                     conv1d_padding, conv1d_dilation, conv1d_groups):
    batch_size, in_channels, length = x.shape
    out_channels, _, kernel_size = conv1d_weight.shape
    
    out_len = (length + 2 * conv1d_padding - conv1d_dilation * (kernel_size - 1) - 1) // conv1d_stride + 1
    output = torch.empty((batch_size, out_channels, out_len), device=x.device, dtype=x.dtype)
    
    conv_ext.conv1d_forward(x, conv1d_weight, conv1d_bias, output, 
                            conv1d_stride, conv1d_padding, conv1d_dilation)
    return output
