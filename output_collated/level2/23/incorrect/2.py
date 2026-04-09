# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260408_235140/code_7.py
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

# -----------------------------------------------------------------------------
# CUDA implementation of fused 3D-Conv + GroupNorm + Mean
# The strategy:
# 1. conv_kernel performs the 3D convolution manually.
# 2. fused_gn_mean_kernel performs GroupNorm and global reduction per batch.
# -----------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void conv3d_naive_kernel(
    const float* __restrict__ input, const float* __restrict__ weight, const float* __restrict__ bias,
    float* __restrict__ output, int N, int C_in, int C_out, int D_in, int H_in, int W_in,
    int D_out, int H_out, int W_out, int KD, int KH, int KW, int stride, int padding, int groups) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C_out * D_out * H_out * W_out;
    if (tid >= total_elements) return;

    int tmp = tid;
    int w = tmp % W_out; tmp /= W_out;
    int h = tmp % H_out; tmp /= H_out;
    int d = tmp % D_out; tmp /= D_out;
    int c = tmp % C_out; tmp /= C_out;
    int n = tmp;

    int g = c / (C_out / groups);
    int c_in_start = g * (C_in / groups);
    int cin_per_g = C_in / groups;

    float acc = 0.0f;
    for (int ci = 0; ci < cin_per_g; ++ci) {
        for (int kd = 0; kd < KD; ++kd) {
            for (int kh = 0; kh < KH; ++kh) {
                for (int kw = 0; kw < KW; ++kw) {
                    int di = d * stride - padding + kd;
                    int hi = h * stride - padding + kh;
                    int wi = w * stride - padding + kw;
                    if (di >= 0 && di < D_in && hi >= 0 && hi < H_in && wi >= 0 && wi < W_in) {
                        int in_idx = ((n * C_in + (c_in_start + ci)) * D_in + di) * H_in * W_in + hi * W_in + wi;
                        int wt_idx = (((c * cin_per_g) + ci) * KD * KH * KW) + kd * KH * KW + kh * KW + kw;
                        acc += input[in_idx] * weight[wt_idx];
                    }
                }
            }
        }
    }
    output[tid] = acc + (bias ? bias[c] : 0.0f);
}

__global__ void groupnorm_mean_kernel(
    const float* __restrict__ x, const float* __restrict__ weight, const float* __restrict__ bias,
    float* __restrict__ res, int N, int C, int D, int H, int W, int num_groups, float eps) {
    
    int n = blockIdx.x;
    int g = blockIdx.y;
    int C_per_g = C / num_groups;
    int total_elements_per_group = C_per_g * D * H * W;
    
    extern __shared__ float smem[];
    float* s_sum = smem;
    float* s_sumsq = &smem[blockDim.x];

    float sum = 0, sumsq = 0;
    for (int i = threadIdx.x; i < total_elements_per_group; i += blockDim.x) {
        int idx = (n * C + (g * C_per_g + (i / (D * H * W)))) * D * H * W + (i % (D * H * W));
        float val = x[idx];
        sum += val;
        sumsq += val * val;
    }

    // Warp reduce
    for (int i = 16; i > 0; i /= 2) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, i);
        sumsq += __shfl_down_sync(0xFFFFFFFF, sumsq, i);
    }
    if (threadIdx.x % 32 == 0) {
        s_sum[threadIdx.x / 32] = sum;
        s_sumsq[threadIdx.x / 32] = sumsq;
    }
    __syncthreads();

    if (threadIdx.x < 32) {
        sum = (threadIdx.x < blockDim.x / 32) ? s_sum[threadIdx.x] : 0;
        sumsq = (threadIdx.x < blockDim.x / 32) ? s_sumsq[threadIdx.x] : 0;
        for (int i = 16; i > 0; i /= 2) {
            sum += __shfl_down_sync(0xFFFFFFFF, sum, i);
            sumsq += __shfl_down_sync(0xFFFFFFFF, sumsq, i);
        }
    }
    
    __shared__ float mean, inv_var;
    if (threadIdx.x == 0) {
        float m = sum / total_elements_per_group;
        mean = m;
        inv_var = rsqrtf((sumsq / total_elements_per_group) - (m * m) + eps);
    }
    __syncthreads();

    float local_batch_sum = 0;
    for (int i = threadIdx.x; i < total_elements_per_group; i += blockDim.x) {
        int c_local = i / (D * H * W);
        int c_global = g * C_per_g + c_local;
        int idx = (n * C + c_global) * D * H * W + (i % (D * H * W));
        float norm = (x[idx] - mean) * inv_var * weight[c_global] + (bias ? bias[c_global] : 0);
        local_batch_sum += norm;
    }

    for (int i = 16; i > 0; i /= 2) local_batch_sum += __shfl_down_sync(0xFFFFFFFF, local_batch_sum, i);
    if (threadIdx.x % 32 == 0) atomicAdd(&res[n], local_batch_sum);
}

void launch_conv(torch::Tensor in, torch::Tensor wt, torch::Tensor bias, torch::Tensor out, int stride, int pad, int groups) {
    int N = in.size(0), C_in = in.size(1), C_out = wt.size(0);
    int D = out.size(2), H = out.size(3), W = out.size(4);
    int total = N * C_out * D * H * W;
    conv3d_naive_kernel<<<(total + 255) / 256, 256>>>(
        in.data_ptr<float>(), wt.data_ptr<float>(), bias.defined() ? bias.data_ptr<float>() : nullptr,
        out.data_ptr<float>(), N, C_in, C_out, in.size(2), in.size(3), in.size(4), D, H, W, wt.size(2), wt.size(3), wt.size(4), stride, pad, groups);
}

void launch_gn(torch::Tensor x, torch::Tensor wt, torch::Tensor bias, torch::Tensor res, int groups, float eps) {
    int N = x.size(0), C = x.size(1), D = x.size(2), H = x.size(3), W = x.size(4);
    dim3 grid(N, groups);
    groupnorm_mean_kernel<<<grid, 256, 512 * sizeof(float)>>>(
        x.data_ptr<float>(), wt.data_ptr<float>(), bias.defined() ? bias.data_ptr<float>() : nullptr,
        res.data_ptr<float>(), N, C, D, H, W, groups, eps);
}
"""

cpp_source = r"""
void launch_conv(torch::Tensor in, torch::Tensor wt, torch::Tensor bias, torch::Tensor out, int stride, int pad, int groups);
void launch_gn(torch::Tensor x, torch::Tensor wt, torch::Tensor bias, torch::Tensor res, int groups, float eps);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_conv", &launch_conv);
    m.def("launch_gn", &launch_gn);
}
"""

fused_ext = load_inline(name='fused_ext', cpp_sources=cpp_source, cuda_sources=cuda_source, extra_cuda_cflags=['-O3'], with_cuda=True)

def functional_model(x, *, conv_weight, conv_bias, conv_stride, conv_padding, conv_dilation, conv_groups, group_norm_weight, group_norm_bias, group_norm_num_groups, group_norm_eps):
    N, C_in, D, H, W = x.shape
    C_out = conv_weight.shape[0]
    D_out = (D + 2 * conv_padding - conv_dilation * (conv_weight.shape[2] - 1) - 1) // conv_stride + 1
    conv_out = torch.empty((N, C_out, D_out, H_out, W_out), device=x.device, dtype=x.dtype)
    fused_ext.launch_conv(x, conv_weight, conv_bias, conv_out, conv_stride, conv_padding, conv_groups)
    res = torch.zeros(N, device=x.device, dtype=torch.float32)
    fused_ext.launch_gn(conv_out, group_norm_weight, group_norm_bias, res, group_norm_num_groups, group_norm_eps)
    return res / (C_out * D_out * H_out * W_out)
