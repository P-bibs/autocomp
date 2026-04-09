# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_084802/code_15.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_weight', 'conv_bias', 'conv_stride', 'conv_padding', 'conv_dilation', 'conv_groups']
REQUIRED_FLAT_STATE_NAMES = ['conv_weight', 'conv_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a convolution, applies minimum operation, Tanh, and another Tanh.
    """

    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

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
    # State for conv (nn.Conv2d)
    if 'conv_weight' in flat_state:
        state_kwargs['conv_weight'] = flat_state['conv_weight']
    else:
        state_kwargs['conv_weight'] = getattr(model.conv, 'weight', None)
    if 'conv_bias' in flat_state:
        state_kwargs['conv_bias'] = flat_state['conv_bias']
    else:
        state_kwargs['conv_bias'] = getattr(model.conv, 'bias', None)
    state_kwargs['conv_stride'] = model.conv.stride
    state_kwargs['conv_padding'] = model.conv.padding
    state_kwargs['conv_dilation'] = model.conv.dilation
    state_kwargs['conv_groups'] = model.conv.groups
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
# CUDA implementation of a direct 2D convolution fused with min reduction and tanh
# This replaces F.conv2d to satisfy the requirement of not using built-in conv functions.
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_conv_min_tanh_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int N, const int C_in, const int H, const int W,
    const int C_out, const int K,
    const int stride_h, const int stride_w,
    const int pad_h, const int pad_w,
    const int dilation_h, const int dilation_w,
    const int out_h, const int out_w)
{
    // Each thread calculates one (n, oh, ow) spatial location
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * out_h * out_w) return;

    int n = idx / (out_h * out_w);
    int rem = idx % (out_h * out_w);
    int oh = rem / out_w;
    int ow = rem % out_w;

    float min_val = 1e38f;

    // Iterate over all output channels
    for (int oc = 0; oc < C_out; ++oc) {
        float sum = (bias != nullptr) ? bias[oc] : 0.0f;

        // Perform spatial convolution for channel oc
        for (int ic = 0; ic < C_in; ++ic) {
            for (int kh = 0; kh < K; ++kh) {
                int i_h = oh * stride_h - pad_h + kh * dilation_h;
                if (i_h < 0 || i_h >= H) continue;
                for (int kw = 0; kw < K; ++kw) {
                    int i_w = ow * stride_w - pad_w + kw * dilation_w;
                    if (i_w < 0 || i_w >= W) continue;

                    float in_val = input[((n * C_in + ic) * H + i_h) * W + i_w];
                    float w_val  = weight[((oc * C_in + ic) * K + kh) * K + kw];
                    sum += in_val * w_val;
                }
            }
        }
        if (oc == 0 || sum < min_val) min_val = sum;
    }

    // Apply tanh twice
    float res = tanhf(min_val);
    res = tanhf(res);

    output[(n * out_h + oh) * out_w + ow] = res;
}

void fused_conv_min_tanh_launcher(
    const torch::Tensor input,
    const torch::Tensor weight,
    const torch::Tensor bias,
    torch::Tensor output,
    const int stride, const int padding, const int dilation)
{
    const int N = input.size(0);
    const int C_in = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);
    const int C_out = weight.size(0);
    const int K = weight.size(2);
    
    int out_h = (H + 2 * padding - dilation * (K - 1) - 1) / stride + 1;
    int out_w = (W + 2 * padding - dilation * (K - 1) - 1) / stride + 1;

    const int total_threads = N * out_h * out_w;
    const int threads = 256;
    const int blocks = (total_threads + threads - 1) / threads;

    fused_conv_min_tanh_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        N, C_in, H, W,
        C_out, K,
        stride, stride,
        padding, padding,
        dilation, dilation,
        out_h, out_w
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_conv_min_tanh_launcher(const torch::Tensor, const torch::Tensor, const torch::Tensor, torch::Tensor, const int, const int, const int);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_conv_min_tanh_launcher, "Fused ConvMinTanh Kernel");
}
"""

fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, conv_weight, conv_bias, conv_stride, conv_padding, conv_dilation, conv_groups):
    # Only support conv_groups=1 based on the logic requested
    assert conv_groups == 1, "Only groups=1 is supported in this fused kernel."
    
    N = x.shape[0]
    C_out = conv_weight.shape[0]
    H, W = x.shape[2], x.shape[3]
    K = conv_weight.shape[2]
    
    out_h = (H + 2 * conv_padding - conv_dilation * (K - 1) - 1) // conv_stride + 1
    out_w = (W + 2 * conv_padding - conv_dilation * (K - 1) - 1) // conv_stride + 1
    
    output = torch.empty((N, 1, out_h, out_w), device=x.device, dtype=x.dtype)
    
    fused_ext.fused_op(x, conv_weight, conv_bias if conv_bias is not None else torch.tensor([]), 
                       output, conv_stride, conv_padding, conv_dilation)
    return output
