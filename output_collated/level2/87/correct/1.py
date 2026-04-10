# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_141042/code_7.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'subtract_value_1', 'subtract_value_2']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_weight', 'conv_bias', 'conv_stride', 'conv_padding', 'conv_dilation', 'conv_groups', 'subtract_value_1', 'subtract_value_2']
REQUIRED_FLAT_STATE_NAMES = ['conv_weight', 'conv_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a convolution, subtracts two values, applies Mish activation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value_1 = subtract_value_1
        self.subtract_value_2 = subtract_value_2

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
    if 'subtract_value_1' in flat_state:
        state_kwargs['subtract_value_1'] = flat_state['subtract_value_1']
    else:
        state_kwargs['subtract_value_1'] = getattr(model, 'subtract_value_1')
    if 'subtract_value_2' in flat_state:
        state_kwargs['subtract_value_2'] = flat_state['subtract_value_2']
    else:
        state_kwargs['subtract_value_2'] = getattr(model, 'subtract_value_2')
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
# CUDA source – Fused convolution, bias, subtraction, and Mish activation
# -------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_conv_mish_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const int N, const int C, const int H, const int W,
    const int outC, const int kH, const int kW,
    const int stride_h, const int stride_w,
    const int pad_h, const int pad_w,
    const int dilation_h, const int dilation_w,
    const int groups,
    const int out_h, const int out_w,
    const float sub1,
    const float sub2,
    float* __restrict__ out)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * outC * out_h * out_w;
    if (idx >= total) return;

    // Decode (n, oc, oh, ow)
    int n = idx / (outC * out_h * out_w);
    int rem = idx % (outC * out_h * out_w);
    int oc = rem / (out_h * out_w);
    int oh = (rem / out_w) % out_h;
    int ow = rem % out_w;

    const int outC_per_group = outC / groups;
    const int inC_per_group = C / groups;
    const int gid = oc / outC_per_group;

    float sum = 0.0f;
    if (bias != nullptr) sum = bias[oc];

    const int start_h = oh * stride_h - pad_h;
    const int start_w = ow * stride_w - pad_w;

    // Manual convolution loop
    for (int ic = 0; ic < inC_per_group; ++ic) {
        int in_channel = gid * inC_per_group + ic;
        for (int kh = 0; kh < kH; ++kh) {
            int h = start_h + kh * dilation_h;
            if (h < 0 || h >= H) continue;
            for (int kw = 0; kw < kW; ++kw) {
                int w = start_w + kw * dilation_w;
                if (w < 0 || w >= W) continue;

                float val = x[((n * C + in_channel) * H + h) * W + w];
                float wgt = weight[(((oc * inC_per_group + ic) * kH) + kh) * kW + kw];
                sum += val * wgt;
            }
        }
    }

    // Apply operations
    sum = sum - sub1 - sub2;
    // Mish: x * tanh(softplus(x))
    float sp = logf(1.0f + expf(sum));
    out[idx] = sum * tanhf(sp);
}

void fused_op_forward(
    const torch::Tensor& x, const torch::Tensor& weight, const torch::Tensor& bias,
    int N, int C, int H, int W, int outC, int kH, int kW,
    int sH, int sW, int pH, int pW, int dH, int dW, int groups,
    int outH, int outW, float sub1, float sub2, torch::Tensor& out)
{
    const int total = N * outC * outH * outW;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    
    fused_conv_mish_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(), weight.data_ptr<float>(), 
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        N, C, H, W, outC, kH, kW, sH, sW, pH, pW, dH, dW, groups, outH, outW, sub1, sub2,
        out.data_ptr<float>()
    );
}
"""

cpp_source = r"""
void fused_op_forward(
    const torch::Tensor& x, const torch::Tensor& weight, const torch::Tensor& bias,
    int N, int C, int H, int W, int outC, int kH, int kW,
    int sH, int sW, int pH, int pW, int dH, int dW, int groups,
    int outH, int outW, float sub1, float sub2, torch::Tensor& out);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused convolution + Mish");
}
"""

fused_ext = load_inline(
    name='fused_op', cpp_sources=cpp_source, cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True
)

def functional_model(x, *, conv_weight, conv_bias, conv_stride, conv_padding, conv_dilation, 
                     conv_groups, subtract_value_1, subtract_value_2):
    N, C, H, W = x.shape
    outC, _, kH, kW = conv_weight.shape
    
    sH, sW = (conv_stride, conv_stride) if isinstance(conv_stride, int) else conv_stride
    pH, pW = (conv_padding, conv_padding) if isinstance(conv_padding, int) else conv_padding
    dH, dW = (conv_dilation, conv_dilation) if isinstance(conv_dilation, int) else conv_dilation
    
    outH = (H + 2 * pH - dH * (kH - 1) - 1) // sH + 1
    outW = (W + 2 * pW - dW * (kW - 1) - 1) // sW + 1
    
    out = torch.empty((N, outC, outH, outW), device=x.device, dtype=x.dtype)
    fused_ext.fused_op(x, conv_weight, conv_bias if conv_bias is not None else torch.tensor([], device=x.device),
                       N, C, H, W, outC, kH, kW, sH, sW, pH, pW, dH, dW, 
                       conv_groups, outH, outW, float(subtract_value_1), float(subtract_value_2), out)
    return out
