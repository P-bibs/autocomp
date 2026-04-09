# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_060810/code_14.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'divisor', 'pool_size', 'bias_shape', 'sum_dim']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_weight', 'conv_bias', 'conv_stride', 'conv_padding', 'conv_dilation', 'conv_groups', 'max_pool_kernel_size', 'max_pool_stride', 'max_pool_padding', 'max_pool_dilation', 'max_pool_ceil_mode', 'max_pool_return_indices', 'global_avg_pool_output_size', 'divisor', 'bias', 'sum_dim']
REQUIRED_FLAT_STATE_NAMES = ['conv_weight', 'conv_bias', 'bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a 3D convolution, divides by a constant, applies max pooling,
    global average pooling, adds a bias term, and sums along a specific dimension.
    """

    def __init__(self, in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.divisor = divisor
        self.max_pool = nn.MaxPool3d(pool_size)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.sum_dim = sum_dim

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
    # State for conv (nn.Conv3d)
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
    # State for max_pool (nn.MaxPool3d)
    state_kwargs['max_pool_kernel_size'] = model.max_pool.kernel_size
    state_kwargs['max_pool_stride'] = model.max_pool.stride
    state_kwargs['max_pool_padding'] = model.max_pool.padding
    state_kwargs['max_pool_dilation'] = model.max_pool.dilation
    state_kwargs['max_pool_ceil_mode'] = model.max_pool.ceil_mode
    state_kwargs['max_pool_return_indices'] = model.max_pool.return_indices
    # State for global_avg_pool (nn.AdaptiveAvgPool3d)
    state_kwargs['global_avg_pool_output_size'] = model.global_avg_pool.output_size
    if 'divisor' in flat_state:
        state_kwargs['divisor'] = flat_state['divisor']
    else:
        state_kwargs['divisor'] = getattr(model, 'divisor')
    if 'bias' in flat_state:
        state_kwargs['bias'] = flat_state['bias']
    else:
        state_kwargs['bias'] = getattr(model, 'bias')
    if 'sum_dim' in flat_state:
        state_kwargs['sum_dim'] = flat_state['sum_dim']
    else:
        state_kwargs['sum_dim'] = getattr(model, 'sum_dim')
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

# =============================================================================
# 1. Custom Conv3d + Fused Reduce CUDA kernels
# =============================================================================
cuda_sources = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C_in, int D_in, int H_in, int W_in,
    int C_out, int Kd, int Kh, int Kw,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int D_out, int H_out, int W_out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_out * D_out * H_out * W_out;
    if (idx >= total) return;

    int tmp = idx;
    int ow = tmp % W_out; tmp /= W_out;
    int oh = tmp % H_out; tmp /= H_out;
    int od = tmp % D_out; tmp /= D_out;
    int oc = tmp % C_out; tmp /= C_out;
    int n  = tmp;

    float val = bias[oc];
    int id_d_base = od * stride_d - pad_d;
    int id_h_base = oh * stride_h - pad_h;
    int id_w_base = ow * stride_w - pad_w;

    for (int cd = 0; cd < Kd; ++cd) {
        int id_d = id_d_base + cd;
        if (id_d < 0 || id_d >= D_in) continue;
        for (int ch = 0; ch < Kh; ++ch) {
            int id_h = id_h_base + ch;
            if (id_h < 0 || id_h >= H_in) continue;
            for (int cw = 0; cw < Kw; ++cw) {
                int id_w = id_w_base + cw;
                if (id_w < 0 || id_w >= W_in) continue;
                
                for (int ic = 0; ic < C_in; ++ic) {
                    val += input[n*(C_in*D_in*H_in*W_in) + ic*(D_in*H_in*W_in) + id_d*(H_in*W_in) + id_h*W_in + id_w] * 
                           weight[oc*(C_in*Kd*Kh*Kw) + ic*(Kd*Kh*Kw) + cd*(Kh*Kw) + ch*Kw + cw];
                }
            }
        }
    }
    output[idx] = val;
}

__global__ void fused_reduce_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    float inv_divisor,
    float sum_bias,
    int N, int C, int D, int H, int W) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * D * H * W;
    if (idx >= total) return;

    int n = idx / (D * H * W);
    int rem = idx % (D * H * W);
    int d = rem / (H * W);
    int h = (rem / W) % H;
    int w = rem % W;

    float sum_val = 0.0f;
    int base_offset = n * (C * D * H * W) + d * (H * W) + h * W + w;
    int channel_stride = D * H * W;

    for (int c = 0; c < C; ++c) {
        sum_val += input[base_offset + c * channel_stride];
    }
    output[idx] = (sum_val * inv_divisor) + sum_bias;
}

void launch_conv3d(torch::Tensor in, torch::Tensor wt, torch::Tensor bias, torch::Tensor out,
                  int stride_d, int stride_h, int stride_w, int pad_d, int pad_h, int pad_w) {
    int N = in.size(0); int C_in = in.size(1);
    int D_in = in.size(2); int H_in = in.size(3); int W_in = in.size(4);
    int C_out = wt.size(0); int Kd = wt.size(2); int Kh = wt.size(3); int Kw = wt.size(4);
    int D_out = out.size(2); int H_out = out.size(3); int W_out = out.size(4);
    int total = N * C_out * D_out * H_out * W_out;
    conv3d_kernel<<<(total + 255)/256, 256>>>(in.data_ptr<float>(), wt.data_ptr<float>(), bias.data_ptr<float>(), out.data_ptr<float>(),
                                              N, C_in, D_in, H_in, W_in, C_out, Kd, Kh, Kw, stride_d, stride_h, stride_w, pad_d, pad_h, pad_w, D_out, H_out, W_out);
}

void launch_fused(torch::Tensor in, torch::Tensor out, float inv_div, float sum_b) {
    int N = in.size(0); int C = in.size(1); int D = in.size(2); int H = in.size(3); int W = in.size(4);
    int total = N * D * H * W;
    fused_reduce_kernel<<<(total + 255)/256, 256>>>(in.data_ptr<float>(), out.data_ptr<float>(), inv_div, sum_b, N, C, D, H, W);
}
"""

cpp_sources = r"""
#include <torch/extension.h>
void launch_conv3d(torch::Tensor in, torch::Tensor wt, torch::Tensor bias, torch::Tensor out, int sd, int sh, int sw, int pd, int ph, int pw);
void launch_fused(torch::Tensor in, torch::Tensor out, float inv_div, float sum_b);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv3d", &launch_conv3d);
    m.def("fused", &launch_fused);
}
"""

ext = load_inline(name='custom_ops', cpp_sources=cpp_sources, cuda_sources=cuda_sources, extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True)

import torch.nn.functional as F

def functional_model(x, *, conv_weight, conv_bias, conv_stride, conv_padding, conv_dilation, conv_groups,
                     max_pool_kernel_size, max_pool_stride, max_pool_padding, max_pool_dilation,
                     max_pool_ceil_mode, max_pool_return_indices, global_avg_pool_output_size,
                     divisor, bias, sum_dim):
    # Custom Conv3d
    N, C_in, D, H, W = x.shape
    C_out = conv_weight.size(0)
    sd, sh, sw = conv_stride
    pd, ph, pw = conv_padding
    D_out = (D + 2*pd - conv_dilation[0]*(conv_weight.size(2)-1) - 1) // sd + 1
    H_out = (H + 2*ph - conv_dilation[1]*(conv_weight.size(3)-1) - 1) // sh + 1
    W_out = (W + 2*pw - conv_dilation[2]*(conv_weight.size(4)-1) - 1) // sw + 1
    out_conv = torch.zeros((N, C_out, D_out, H_out, W_out), device=x.device)
    ext.conv3d(x.contiguous(), conv_weight.contiguous(), conv_bias.contiguous(), out_conv, sd, sh, sw, pd, ph, pw)
    
    # Built-in Poolings
    x = F.max_pool3d(out_conv, max_pool_kernel_size, max_pool_stride, max_pool_padding, max_pool_dilation, max_pool_ceil_mode, max_pool_return_indices)
    x = F.adaptive_avg_pool3d(x, global_avg_pool_output_size)
    
    # Custom Fused Reduce
    out_final = torch.zeros((x.size(0), x.size(2), x.size(3), x.size(4)), device=x.device)
    ext.fused(x.contiguous(), out_final, 1.0/divisor, bias.sum().item())
    return out_final
