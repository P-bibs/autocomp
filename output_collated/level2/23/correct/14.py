# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_000325/code_22.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'num_groups']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_weight', 'conv_bias', 'conv_stride', 'conv_padding', 'conv_dilation', 'conv_groups', 'group_norm_weight', 'group_norm_bias', 'group_norm_num_groups', 'group_norm_eps']
REQUIRED_FLAT_STATE_NAMES = ['conv_weight', 'conv_bias', 'group_norm_weight', 'group_norm_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a 3D convolution, applies Group Normalization, computes the mean
    """

    def __init__(self, in_channels, out_channels, kernel_size, num_groups):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)

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
    # State for group_norm (nn.GroupNorm)
    if 'group_norm_weight' in flat_state:
        state_kwargs['group_norm_weight'] = flat_state['group_norm_weight']
    else:
        state_kwargs['group_norm_weight'] = getattr(model.group_norm, 'weight', None)
    if 'group_norm_bias' in flat_state:
        state_kwargs['group_norm_bias'] = flat_state['group_norm_bias']
    else:
        state_kwargs['group_norm_bias'] = getattr(model.group_norm, 'bias', None)
    state_kwargs['group_norm_num_groups'] = model.group_norm.num_groups
    state_kwargs['group_norm_eps'] = model.group_norm.eps
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
# Optimized CUDA kernels:
# 1. Custom 3D Conv implementation (no ATen convolution)
# 2. Fused GroupNorm + Weighted Reduction with Grid-Stride loops
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void conv3d_naive_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int inC, int D, int H, int W,
    int outC, int outD, int outH, int outW,
    int kD, int kH, int kW,
    int groups)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * outC * outD * outH * outW;
    if (tid >= total_elements) return;

    int n = tid / (outC * outD * outH * outW);
    int rem = tid % (outC * outD * outH * outW);
    int oc = rem / (outD * outH * outW);
    rem %= (outD * outH * outW);
    int od = rem / (outH * outW);
    rem %= (outH * outW);
    int oh = rem / outW;
    int ow = rem % outW;

    int inC_per_g = inC / groups;
    int g = oc / (outC / groups);
    int inC_start = g * inC_per_g;

    float acc = (bias != nullptr) ? bias[oc] : 0.0f;

    for (int kd = 0; kd < kD; ++kd) {
        int id = od + kd;
        for (int kh = 0; kh < kH; ++kh) {
            int ih = oh + kh;
            for (int kw = 0; kw < kW; ++kw) {
                int iw = ow + kw;
                for (int ic = 0; ic < inC_per_g; ++ic) {
                    int input_ic = inC_start + ic;
                    int weight_idx = (oc * inC_per_g + ic) * (kD * kH * kW) + (kd * kH * kW + kh * kW + kw);
                    int input_idx = (((n * inC + input_ic) * D + id) * H + ih) * W + iw;
                    acc += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    output[tid] = acc;
}

__global__ void fused_norm_reduce_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C, int D, int H, int W, int G, float eps)
{
    extern __shared__ float s[];
    float* s_sum = s;
    float* s_sum2 = &s[blockDim.x];

    int ng = blockIdx.x; 
    int n = ng / G;
    int g = ng % G;

    int C_per_G = C / G;
    int spatial = D * H * W;
    int group_size = C_per_G * spatial;
    int c_offset = g * C_per_G;

    float p_sum = 0.0f, p_sum2 = 0.0f;
    for (int idx = threadIdx.x; idx < group_size; idx += blockDim.x) {
        int c = c_offset + (idx / spatial);
        int s_idx = idx % spatial;
        int d = s_idx / (H * W);
        int rem = s_idx % (H * W);
        int h = rem / W;
        int w = rem % W;
        float val = input[(((n * C + c) * D + d) * H + h) * W + w];
        p_sum += val;
        p_sum2 += val * val;
    }

    s_sum[threadIdx.x] = p_sum;
    s_sum2[threadIdx.x] = p_sum2;
    __syncthreads();

    for (int s_red = blockDim.x / 2; s_red > 0; s_red >>= 1) {
        if (threadIdx.x < s_red) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + s_red];
            s_sum2[threadIdx.x] += s_sum2[threadIdx.x + s_red];
        }
        __syncthreads();
    }

    float mean = s_sum[0] / group_size;
    float var = (s_sum2[0] / group_size) - (mean * mean);
    float inv_std = rsqrtf(var + eps);

    float p_out = 0.0f;
    for (int idx = threadIdx.x; idx < group_size; idx += blockDim.x) {
        int c = c_offset + (idx / spatial);
        int s_idx = idx % spatial;
        int d = s_idx / (H * W), h = (s_idx / W) % H, w = s_idx % W;
        float val = input[(((n * C + c) * D + d) * H + h) * W + w];
        p_out += ((val - mean) * inv_std * weight[c] + bias[c]);
    }
    s_sum[threadIdx.x] = p_out;
    __syncthreads();
    for (int s_red = blockDim.x / 2; s_red > 0; s_red >>= 1) {
        if (threadIdx.x < s_red) s_sum[threadIdx.x] += s_sum[threadIdx.x + s_red];
        __syncthreads();
    }
    if (threadIdx.x == 0) output[ng] = s_sum[0];
}

void launch_conv3d(torch::Tensor in, torch::Tensor wt, torch::Tensor bs, torch::Tensor ot, int N, int C, int D, int H, int W, int oC, int oD, int oH, int oW, int kD, int kH, int kW, int G) {
    int total = N * oC * oD * oH * oW;
    conv3d_naive_kernel<<<(total + 255) / 256, 256>>>(in.data_ptr<float>(), wt.data_ptr<float>(), bs.defined() ? bs.data_ptr<float>() : nullptr, ot.data_ptr<float>(), N, C, D, H, W, oC, oD, oH, oW, kD, kH, kW, G);
}

void launch_fused(torch::Tensor in, torch::Tensor wt, torch::Tensor bs, torch::Tensor ot, int N, int C, int D, int H, int W, int G, float eps) {
    fused_norm_reduce_kernel<<<N * G, 256, 512 * sizeof(float)>>>(in.data_ptr<float>(), wt.data_ptr<float>(), bs.data_ptr<float>(), ot.data_ptr<float>(), N, C, D, H, W, G, eps);
}
"""

cpp_source = r"""
void launch_conv3d(torch::Tensor in, torch::Tensor wt, torch::Tensor bs, torch::Tensor ot, int N, int C, int D, int H, int W, int oC, int oD, int oH, int oW, int kD, int kH, int kW, int G);
void launch_fused(torch::Tensor in, torch::Tensor wt, torch::Tensor bs, torch::Tensor ot, int N, int C, int D, int H, int W, int G, float eps);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv3d", &launch_conv3d);
    m.def("fused", &launch_fused);
}
"""
ext = load_inline('fused_ext', cpp_source, cuda_source, with_cuda=True, extra_cuda_cflags=['-O3'])

def functional_model(x, *, conv_weight, conv_bias, conv_stride, conv_padding, conv_dilation, conv_groups, group_norm_weight, group_norm_bias, group_norm_num_groups, group_norm_eps):
    N, C, D, H, W = x.shape
    kD, kH, kW = conv_weight.shape[2:]
    oD, oH, oW = (D+2*conv_padding[0]-conv_dilation[0]*(kD-1)-1)//conv_stride[0]+1, (H+2*conv_padding[1]-conv_dilation[1]*(kH-1)-1)//conv_stride[1]+1, (W+2*conv_padding[2]-conv_dilation[2]*(kW-1)-1)//conv_stride[2]+1
    out = torch.empty((N, conv_weight.size(0), oD, oH, oW), device=x.device)
    ext.conv3d(x, conv_weight, conv_bias, out, N, C, D, H, W, out.size(1), oD, oH, oW, kD, kH, kW, conv_groups)
    group_sums = torch.empty(N * group_norm_num_groups, device=x.device)
    ext.fused(out, group_norm_weight, group_norm_bias, group_sums, N, out.size(1), oD, oH, oW, group_norm_num_groups, group_norm_eps)
    return group_sums.view(N, group_norm_num_groups).sum(dim=1) / (out.size(1) * oD * oH * oW)
