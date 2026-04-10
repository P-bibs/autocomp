# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_071708/code_5.py
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

# Optimization: Implemented custom 2D Convolution Kernel in CUDA
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv2d_kernel(const float* __restrict__ input, const float* __restrict__ weight, 
                              float* __restrict__ output, int N, int C, int H, int W, 
                              int O, int kH, int kW, int stride, int padding) {
    int o = blockIdx.x;
    int n = blockIdx.y;
    int h_out = blockIdx.z / (W / stride);
    int w_out = blockIdx.z % (W / stride);

    float acc = 0.0f;
    for (int c = 0; c < C; ++c) {
        for (int i = 0; i < kH; ++i) {
            for (int j = 0; j < kW; ++j) {
                int h_in = h_out * stride + i - padding;
                int w_in = w_out * stride + j - padding;
                if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                    acc += input[((n * C + c) * H + h_in) * W + w_in] * 
                           weight[((o * C + c) * kH + i) * kW + j];
                }
            }
        }
    }
    output[((n * O + o) * (H / stride) + h_out) * (W / stride) + w_out] = acc;
}

void conv2d_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor output, 
                    int stride, int padding) {
    int N = input.size(0), C = input.size(1), H = input.size(2), W = input.size(3);
    int O = weight.size(0), kH = weight.size(2), kW = weight.size(3);
    dim3 blocks(O, N, (H / stride) * (W / stride));
    conv2d_kernel<<<blocks, 1>>>(input.data_ptr<float>(), weight.data_ptr<float>(), 
                                 output.data_ptr<float>(), N, C, H, W, O, kH, kW, stride, padding);
}
"""

cpp_source = r"""
void conv2d_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor output, int stride, int padding);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv2d_forward", &conv2d_forward, "Custom 2D Convolution Forward");
}
"""

# Compile the extension
conv_ext = load_inline(
    name='custom_conv',
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
    # Enforce basic constraints for the custom kernel implementation
    assert conv1d_dilation == 1 and conv1d_groups == 1, "Native kernel optimized for standard Conv2D"
    N, C, H, W = x.shape
    O, _, kH, kW = conv1d_weight.shape
    
    out = torch.zeros((N, O, H // conv1d_stride, W // conv1d_stride), device=x.device)
    conv_ext.conv2d_forward(x.contiguous(), conv1d_weight.contiguous(), out, conv1d_stride, conv1d_padding)
    
    return out + conv1d_bias.view(1, -1, 1, 1)
