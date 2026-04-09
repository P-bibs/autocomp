# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_052229/code_7.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_weight', 'conv_bias', 'conv_stride', 'conv_padding', 'conv_dilation', 'conv_groups']
REQUIRED_FLAT_STATE_NAMES = ['conv_weight', 'conv_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a convolution, applies HardSwish, and then ReLU.
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
# Inline CUDA source – Fused convolution, hardswish, and relu kernel.
# Optimized to process output elements in parallel.
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void fused_conv_hardswish_relu_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int N, const int C_in, const int C_out,
    const int H_in, const int W_in,
    const int K,
    const int stride_h, const int stride_w,
    const int pad_h, const int pad_w,
    const int dilation_h, const int dilation_w,
    const int groups,
    const int H_out, const int W_out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_out = N * C_out * H_out * W_out;
    if (idx >= total_out) return;

    // Unpack linear index to 4D coordinates
    int tmp = idx;
    int ow = tmp % W_out; tmp /= W_out;
    int oh = tmp % H_out; tmp /= H_out;
    int oc = tmp % C_out; tmp /= C_out;
    int n  = tmp;

    // Grouped convolution logic
    const int c_out_per_group = C_out / groups;
    const int c_in_per_group  = C_in / groups;
    const int group_id = oc / c_out_per_group;
    const int ic_base = group_id * c_in_per_group;

    float sum = 0.0f;
    for (int ic = 0; ic < c_in_per_group; ++ic) {
        int input_c = ic_base + ic;
        for (int kh = 0; kh < K; ++kh) {
            int ih = oh * stride_h + kh * dilation_h - pad_h;
            if (ih < 0 || ih >= H_in) continue;
            for (int kw = 0; kw < K; ++kw) {
                int iw = ow * stride_w + kw * dilation_w - pad_w;
                if (iw < 0 || iw >= W_in) continue;

                // Weight layout: [C_out, C_in/groups, K, K]
                int w_idx = (oc * c_in_per_group + ic) * (K * K) + kh * K + kw;
                int i_idx = ((n * C_in + input_c) * H_in + ih) * W_in + iw;
                sum += input[i_idx] * weight[w_idx];
            }
        }
    }

    if (bias) sum += bias[oc];

    // Activation: Hardswish(x) = x * ReLU6(x + 3) / 6
    float hs = sum * fminf(fmaxf(sum + 3.0f, 0.0f), 6.0f) * 0.16666667f;
    // ReLU activation: max(0, hs)
    output[idx] = fmaxf(hs, 0.0f);
}

void fused_conv(
    at::Tensor input, at::Tensor weight, at::Tensor bias, at::Tensor output,
    int N, int C_in, int C_out, int H_in, int W_in, int K,
    int stride_h, int stride_w, int pad_h, int pad_w,
    int dilation_h, int dilation_w, int groups, int H_out, int W_out)
{
    const int total_out = N * C_out * H_out * W_out;
    const int threads = 256;
    const int blocks = (total_out + threads - 1) / threads;

    fused_conv_hardswish_relu_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        N, C_in, C_out, H_in, W_in, K,
        stride_h, stride_w, pad_h, pad_w,
        dilation_h, dilation_w, groups, H_out, W_out);
}
"""

cpp_source = r"""
void fused_conv(at::Tensor input, at::Tensor weight, at::Tensor bias, at::Tensor output,
                int N, int C_in, int C_out, int H_in, int W_in, int K,
                int stride_h, int stride_w, int pad_h, int pad_w,
                int dilation_h, int dilation_w, int groups, int H_out, int W_out);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv", &fused_conv, "Fused convolution activation");
}
"""

fused_ext = load_inline(
    name='fused_module',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, conv_weight, conv_bias, conv_stride, conv_padding, conv_dilation, conv_groups):
    x = x.contiguous()
    N, C_in, H_in, W_in = x.shape
    C_out = conv_weight.shape[0]
    K = conv_weight.shape[2]
    
    stride = (conv_stride, conv_stride) if isinstance(conv_stride, int) else conv_stride
    pad = (conv_padding, conv_padding) if isinstance(conv_padding, int) else conv_padding
    dil = (conv_dilation, conv_dilation) if isinstance(conv_dilation, int) else conv_dilation
    
    H_out = (H_in + 2 * pad[0] - dil[0] * (K - 1) - 1) // stride[0] + 1
    W_out = (W_in + 2 * pad[1] - dil[1] * (K - 1) - 1) // stride[1] + 1
    
    output = torch.empty((N, C_out, H_out, W_out), device=x.device, dtype=x.dtype)
    
    fused_ext.fused_conv(
        x, conv_weight, conv_bias if conv_bias is not None else torch.tensor([], device=x.device), output,
        N, C_in, C_out, H_in, W_in, K,
        stride[0], stride[1], pad[0], pad[1], dil[0], dil[1], conv_groups, H_out, W_out
    )
    return output
