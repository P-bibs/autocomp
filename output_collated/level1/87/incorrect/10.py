# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_070444/code_5.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'bias']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv1d_weight', 'conv1d_bias', 'conv1d_stride', 'conv1d_padding', 'conv1d_dilation', 'conv1d_groups']
REQUIRED_FLAT_STATE_NAMES = ['conv1d_weight', 'conv1d_bias']


class ModelNew(nn.Module):
    """
    Performs a pointwise 2D convolution operation.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """

    def __init__(self, in_channels: int, out_channels: int, bias: bool=False):
        super(ModelNew, self).__init__()
        self.conv1d = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)

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
    # State for conv1d (nn.Conv2d)
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

# Optimization: Direct window-based 2D convolution kernel.
# We focus on a simple block-based access pattern.
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int B, int C_in, int C_out, int H, int W, int K) {
    
    int n = blockIdx.z;
    int co = blockIdx.y;
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    int w = blockIdx.y * blockDim.y + threadIdx.y;
    
    int out_H = H - K + 1;
    int out_W = W - K + 1;

    if (h < out_H && w < out_W) {
        float val = bias[co];
        for (int ci = 0; ci < C_in; ++ci) {
            for (int kh = 0; kh < K; ++kh) {
                for (int kw = 0; kw < K; ++kw) {
                    val += input[n * (C_in * H * W) + ci * (H * W) + (h + kh) * W + (w + kw)] *
                           weight[co * (C_in * K * K) + ci * (K * K) + kh * K + kw];
                }
            }
        }
        output[n * (C_out * out_H * out_W) + co * (out_H * out_W) + h * out_W + w] = val;
    }
}

torch::Tensor conv2d_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    int B = input.size(0);
    int C_in = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    int C_out = weight.size(0);
    int K = weight.size(2);
    
    auto output = torch::zeros({B, C_out, H - K + 1, W - K + 1}, input.options());
    
    dim3 threads(16, 16);
    dim3 blocks((H - K + 1 + 15) / 16, (W - K + 1 + 15) / 16, B);
    
    conv2d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), B, C_in, C_out, H, W, K
    );
    return output;
}
"""

cpp_source = r"""
#include <torch/extension.h>
torch::Tensor conv2d_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv2d_cuda", &conv2d_cuda, "Custom 2D Convolution Kernel");
}
"""

# Compile the extension
conv_ext = load_inline(
    name='conv2d_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, conv1d_weight, conv1d_bias, conv1d_stride, conv1d_padding, conv1d_dilation, conv1d_groups):
    # Specialized CUDA wrapper for 2D convolution via the compiled extension
    # Note: Logic assumes K=3 or similar fixed kernel sizes typical for this structure
    return conv_ext.conv2d_cuda(x, conv1d_weight, conv1d_bias)

# Variables defined as per the requirements
batch_size = 16
in_channels = 64
out_channels = 128
width = 1024
height = 1024
