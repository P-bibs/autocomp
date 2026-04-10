# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_071251/code_5.py
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

# Optimization Strategy:
# We implement a custom CUDA kernel that performs a direct 2D convolution.
# To maximize performance on the RTX 2080Ti, we map each output pixel (b, oc, oh, ow) 
# to a single CUDA thread. This reduces shared memory bank conflicts compared to 
# complex tiling schemes for large kernels and leverages the L1/L2 cache 
# hierarchy for weights.

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void conv2d_kernel(const float* __restrict__ input, 
                              const float* __restrict__ weight, 
                              const float* __restrict__ bias,
                              float* __restrict__ output, 
                              int B, int IC, int IH, int IW, 
                              int OC, int KH, int KW) {
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y;
    int oc = blockIdx.z / B;
    int b  = blockIdx.z % B;

    if (ow >= IW || oh >= IH) return;

    float sum = 0.0f;
    // Iterate over channels and kernel spatial dimensions
    for (int ic = 0; ic < IC; ++ic) {
        for (int kh = 0; kh < KH; ++kh) {
            for (int kw = 0; kw < KW; ++kw) {
                int ih = oh + kh - KH/2; // Assuming kernel center padding for example
                int iw = ow + kw - KW/2;
                
                if (ih >= 0 && ih < IH && iw >= 0 && iw < IW) {
                    sum += input[((b * IC + ic) * IH + ih) * IW + iw] * 
                           weight[((oc * IC + ic) * KH + kh) * KW + kw];
                }
            }
        }
    }
    
    int out_idx = ((b * OC + oc) * IH + oh) * IW + ow;
    output[out_idx] = sum + bias[oc];
}

void launch_conv2d(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output) {
    const int B = input.size(0);
    const int IC = input.size(1);
    const int IH = input.size(2);
    const int IW = input.size(3);
    const int OC = weight.size(0);
    const int KH = weight.size(2);
    const int KW = weight.size(3);

    dim3 threads(32);
    dim3 blocks((IW + 31) / 32, IH, B * OC);
    
    conv2d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), B, IC, IH, IW, OC, KH, KW
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void launch_conv2d(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_conv", &launch_conv2d, "Custom CUDA Conv2d");
}
"""

conv_ext = load_inline(
    name='conv_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, conv1d_weight, conv1d_bias, conv1d_stride, conv1d_padding, conv1d_dilation, conv1d_groups):
    # Output shape: [Batch, OutChannels, Height, Width]
    out = torch.empty((x.size(0), conv1d_weight.size(0), x.size(2), x.size(3)), device='cuda')
    conv_ext.launch_conv(x, conv1d_weight, conv1d_bias, out)
    return out

# Global testing parameters as per requirements
batch_size, in_channels, out_channels, width, height = 16, 64, 128, 1024, 1024

def get_init_inputs():
    return [in_channels, out_channels]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width, device='cuda')]
