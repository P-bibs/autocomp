# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260408_235554/code_7.py
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

# -------------------------------------------------------------------------
# CUDA source – custom 3D convolution and fused GroupNorm+Mean kernel.
# -------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv3d_naive_kernel(
    const float* __restrict__ input, const float* __restrict__ weight, const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C_in, int D_in, int H_in, int W_in,
    int C_out, int D_out, int H_out, int W_out,
    int Kd, int Kh, int Kw,
    int str_d, int str_h, int str_w,
    int pad_d, int pad_h, int pad_w,
    int dil_d, int dil_h, int dil_w,
    int groups) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int numel = N * C_out * D_out * H_out * W_out;
    if (idx >= numel) return;

    int tmp = idx;
    int w = tmp % W_out; tmp /= W_out;
    int h = tmp % H_out; tmp /= H_out;
    int d = tmp % D_out; tmp /= D_out;
    int c = tmp % C_out; tmp /= C_out;
    int n = tmp;

    int c_in_groups = C_in / groups;
    int group_id = c / (C_out / groups);
    int c_in_start = group_id * c_in_groups;

    float acc = bias ? bias[c] : 0.0f;
    int in_d_base = d * str_d - pad_d;
    int in_h_base = h * str_h - pad_h;
    int in_w_base = w * str_w - pad_w;

    for (int ci = 0; ci < c_in_groups; ++ci) {
        int actual_ci = c_in_start + ci;
        for (int kd = 0; kd < Kd; ++kd) {
            int di = in_d_base + kd * dil_d;
            if (di < 0 || di >= D_in) continue;
            for (int kh = 0; kh < Kh; ++kh) {
                int hi = in_h_base + kh * dil_h;
                if (hi < 0 || hi >= H_in) continue;
                for (int kw = 0; kw < Kw; ++kw) {
                    int wi = in_w_base + kw * dil_w;
                    if (wi < 0 || wi >= W_in) continue;
                    
                    float in_val = input[((n * C_in + actual_ci) * D_in + di) * H_in * W_in + hi * W_in + wi];
                    float w_val = weight[(((c * c_in_groups + ci) * Kd + kd) * Kh + kh) * Kw + kw];
                    acc += in_val * w_val;
                }
            }
        }
    }
    output[idx] = acc;
}

__global__ void fused_norm_mean_kernel(
    const float* __restrict__ x, const float* __restrict__ weight, const float* __restrict__ bias,
    float* __restrict__ out,
    int N, int C, int D, int H, int W, int num_groups, float eps)
{
    extern __shared__ float sdata[];
    int n = blockIdx.x;
    int group_size = C / num_groups;
    int elements_per_group = group_size * D * H * W;
    int total_elements = C * D * H * W;

    float* g_mean = sdata;
    float* g_invstd = sdata + num_groups;
    
    // 1. Calculate Per-Group Mean and Var
    for(int g = threadIdx.x; g < num_groups; g += blockDim.x) {
        float sum = 0.0f; float sq_sum = 0.0f;
        for(int c = g * group_size; c < (g+1) * group_size; ++c) {
            for(int i = 0; i < D*H*W; ++i) {
                float val = x[((n * C + c) * D * H * W) + i];
                sum += val; sq_sum += val * val;
            }
        }
        float mean = sum / elements_per_group;
        g_mean[g] = mean;
        g_invstd[g] = rsqrtf((sq_sum / elements_per_group) - (mean * mean) + eps);
    }
    __syncthreads();

    // 2. Normalize and compute mean of batch item
    float batch_sum = 0.0f;
    for(int i = threadIdx.x; i < total_elements; i += blockDim.x) {
        int c = i / (D * H * W);
        int g = c / group_size;
        float val = x[n * total_elements + i];
        float norm = (val - g_mean[g]) * g_invstd[g] * weight[c] + bias[c];
        batch_sum += norm;
    }
    
    // 3. Block-wide reduction
    float* red = sdata + 2 * num_groups;
    red[threadIdx.x] = batch_sum;
    for(int s = blockDim.x >> 1; s > 0; s >>= 1) {
        __syncthreads();
        if(threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
    }
    if(threadIdx.x == 0) out[n] = red[0] / total_elements;
}

void launch_conv3d(at::Tensor in, at::Tensor w, at::Tensor b, at::Tensor out, int N, int C_in, int D_in, int H_in, int W_in, int C_out, int D_out, int H_out, int W_out, int Kd, int Kh, int Kw, int sd, int sh, int sw, int pd, int ph, int pw, int dd, int dh, int dw, int groups) {
    int numel = N * C_out * D_out * H_out * W_out;
    int threads = 256;
    conv3d_naive_kernel<<<(numel + threads - 1) / threads, threads>>> (in.data_ptr<float>(), w.data_ptr<float>(), b.defined() ? b.data_ptr<float>() : nullptr, out.data_ptr<float>(), N, C_in, D_in, H_in, W_in, C_out, D_out, H_out, W_out, Kd, Kh, Kw, sd, sh, sw, pd, ph, pw, dd, dh, dw, groups);
}

void launch_fused(at::Tensor x, at::Tensor w, at::Tensor b, at::Tensor out, int N, int C, int D, int H, int W, int num_groups, float eps) {
    int threads = 256;
    size_t smem = (2 * num_groups + threads) * sizeof(float);
    fused_norm_mean_kernel<<<N, threads, smem>>>(x.data_ptr<float>(), w.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), N, C, D, H, W, num_groups, eps);
}
"""

cpp_source = """
void launch_conv3d(at::Tensor in, at::Tensor w, at::Tensor b, at::Tensor out, int N, int Ci, int Di, int Hi, int Wi, int Co, int Do, int Ho, int Wo, int Kd, int Kh, int Kw, int sd, int sh, int sw, int pd, int ph, int pw, int dd, int dh, int dw, int groups);
void launch_fused(at::Tensor x, at::Tensor w, at::Tensor b, at::Tensor out, int N, int C, int D, int H, int W, int num_groups, float eps);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv3d", &launch_conv3d);
    m.def("fused", &launch_fused);
}
"""

mod = load_inline(name='ops', cpp_sources=cpp_source, cuda_sources=cuda_source, with_cuda=True, extra_cuda_cflags=['-O3', '--use_fast_math'])

def functional_model(x, *, conv_weight, conv_bias, conv_stride, conv_padding, conv_dilation, conv_groups, group_norm_weight, group_norm_bias, group_norm_num_groups, group_norm_eps):
    N, Ci, Di, Hi, Wi = x.shape
    Co, _, Kd, Kh, Kw = conv_weight.shape
    sd, sh, sw = (conv_stride,)*3 if isinstance(conv_stride, int) else conv_stride
    pd, ph, pw = (conv_padding,)*3 if isinstance(conv_padding, int) else conv_padding
    dd, dh, dw = (conv_dilation,)*3 if isinstance(conv_dilation, int) else conv_dilation
    Do = (Di + 2*pd - dd*(Kd-1) - 1) // sd + 1
    Ho = (Hi + 2*ph - dh*(Kh-1) - 1) // sh + 1
    Wo = (Wi + 2*pw - dw*(Kw-1) - 1) // sw + 1
    out_conv = torch.empty((N, Co, Do, Ho, Wo), device='cuda')
    mod.conv3d(x, conv_weight, conv_bias, out_conv, N, Ci, Di, Hi, Wi, Co, Do, Ho, Wo, Kd, Kh, Kw, sd, sh, sw, pd, ph, pw, dd, dh, dw, conv_groups)
    out = torch.empty((N,), device='cuda')
    mod.fused(out_conv, group_norm_weight, group_norm_bias, out, N, Co, Do, Ho, Wo, group_norm_num_groups, group_norm_eps)
    return out
