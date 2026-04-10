# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260410_065229/code_5.py
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

# CUDA Kernel: Implicit GEMM structure
# This kernel maps the 2D Convolution to a Matrix Multiplication operation.
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void conv2d_implicit_gemm_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int B, int C_in, int H, int W,
    int C_out, int K) {

    int out_c = blockIdx.x;
    int b = blockIdx.y;
    int h = threadIdx.y;
    int w = threadIdx.x;

    if (out_c >= C_out || b >= B || h >= (H - K + 1) || w >= (W - K + 1)) return;

    float sum = bias[out_c];
    for (int kh = 0; kh < K; ++kh) {
        for (int kw = 0; kw < K; ++kw) {
            for (int cin = 0; cin < C_in; ++cin) {
                int x_idx = ((b * C_in + cin) * H + (h + kh)) * W + (w + kw);
                int w_idx = ((out_c * C_in + cin) * K + kh) * K + kw;
                sum += x[x_idx] * weight[w_idx];
            }
        }
    }
    out[((b * C_out + out_c) * (H - K + 1) + h) * (W - K + 1) + w] = sum;
}

void fused_op_dispatch(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, torch::Tensor out) {
    int B = x.size(0);
    int C_in = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    int C_out = weight.size(0);
    int K = weight.size(2);

    dim3 threads(W - K + 1, H - K + 1);
    dim3 blocks(C_out, B);

    conv2d_implicit_gemm_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        out.data_ptr<float>(), B, C_in, H, W, C_out, K
    );
}
"""

cpp_source = r"""
void fused_op_dispatch(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, torch::Tensor out);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_dispatch, "Implicit GEMM Conv2D");
}
"""

fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, conv1d_weight, conv1d_bias, conv1d_stride, conv1d_padding, conv1d_dilation, conv1d_groups):
    # Ensure inputs are contiguous on GPU
    x = x.contiguous()
    weight = conv1d_weight.contiguous()
    bias = conv1d_bias.contiguous()
    
    # Calculate output dimensions
    batch, _, height, width = x.shape
    out_c, _, k_h, k_w = weight.shape
    
    h_out = height - k_h + 1
    w_out = width - k_w + 1
    
    out = torch.empty((batch, out_c, h_out, w_out), device=x.device, dtype=x.dtype)
    
    # Dispatch custom kernel
    fused_ext.fused_op(x, weight, bias, out)
    return out
