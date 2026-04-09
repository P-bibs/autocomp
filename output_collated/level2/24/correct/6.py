# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_101218/code_7.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'dim']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_weight', 'conv_bias', 'conv_stride', 'conv_padding', 'conv_dilation', 'conv_groups', 'dim']
REQUIRED_FLAT_STATE_NAMES = ['conv_weight', 'conv_bias']


class ModelNew(nn.Module):
    """
    Simple model that performs a 3D convolution, applies minimum operation along a specific dimension, 
    and then applies softmax.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dim):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.dim = dim

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
    if 'dim' in flat_state:
        state_kwargs['dim'] = flat_state['dim']
    else:
        state_kwargs['dim'] = getattr(model, 'dim')
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
# CUDA kernel – fused 3-D convolution + depth-wise min + channel-softmax
# -------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

__global__ void fused_conv_min_softmax_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ out,
    const int N, const int C_in, const int D_in, const int H_in, const int W_in,
    const int C_out,
    const int Kd, const int Kh, const int Kw,
    const int stride_d, const int stride_h, const int stride_w,
    const int pad_d, const int pad_h, const int pad_w,
    const int dil_d, const int dil_h, const int dil_w,
    const int groups,
    const int D_out, const int H_out, const int W_out)
{
    extern __shared__ float s_data[];
    int tid = threadIdx.x;
    int spatial_id = blockIdx.x;
    
    int w = spatial_id % W_out;
    int tmp = spatial_id / W_out;
    int h = tmp % H_out;
    int n = tmp / H_out;

    int C_per_group = C_out / groups;
    int Cin_per_group = C_in / groups;

    // Per-channel output calculation
    if (tid < C_out) {
        int group_idx = tid / C_per_group;
        float bias_val = (bias != nullptr) ? bias[tid] : 0.0f;
        float min_val = FLT_MAX;

        for (int d_out = 0; d_out < D_out; ++d_out) {
            float sum = bias_val;
            for (int ci = 0; ci < Cin_per_group; ++ci) {
                int c_in = group_idx * Cin_per_group + ci;
                for (int kd = 0; kd < Kd; ++kd) {
                    int d_in = d_out * stride_d - pad_d + kd * dil_d;
                    if (d_in < 0 || d_in >= D_in) continue;
                    for (int kh = 0; kh < Kh; ++kh) {
                        int h_in = h * stride_h - pad_h + kh * dil_h;
                        if (h_in < 0 || h_in >= H_in) continue;
                        for (int kw = 0; kw < Kw; ++kw) {
                            int w_in = w * stride_w - pad_w + kw * dil_w;
                            if (w_in < 0 || w_in >= W_in) continue;
                            
                            int in_idx = ((n * C_in + c_in) * D_in + d_in) * H_in * W_in + h_in * W_in + w_in;
                            int w_idx = (((tid) * Cin_per_group + ci) * Kd + kd) * Kh * Kw + kh * Kw + kw;
                            sum += x[in_idx] * weight[w_idx];
                        }
                    }
                }
            }
            if (sum < min_val) min_val = sum;
        }
        s_data[tid] = min_val;
    }
    __syncthreads();

    // Softmax logic
    if (tid < C_out) {
        float max_val = s_data[0];
        for(int i=1; i<C_out; ++i) if(s_data[i] > max_val) max_val = s_data[i];
        
        float val = __expf(s_data[tid] - max_val);
        s_data[tid + C_out] = val;
        __syncthreads();
        
        float sum = 0.0f;
        for(int i=0; i<C_out; ++i) sum += s_data[i + C_out];
        
        int out_idx = ((n * C_out + tid) * H_out + h) * W_out + w;
        out[out_idx] = val / sum;
    }
}
"""

cpp_source = r"""
#include <torch/extension.h>
void launch_fused(const torch::Tensor& x, const torch::Tensor& weight, const torch::Tensor& bias, torch::Tensor& out, 
                  int N, int C_in, int D_in, int H_in, int W_in, int C_out, int Kd, int Kh, int Kw, 
                  int sd, int sh, int sw, int pd, int ph, int pw, int dd, int dh, int dw, int groups,
                  int D_out, int H_out, int W_out, int smem);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("fused_op", &launch_fused); }
"""

cuda_wrapper = r"""
#include <torch/extension.h>
void launch_fused(const torch::Tensor& x, const torch::Tensor& weight, const torch::Tensor& bias, torch::Tensor& out, 
                  int N, int C_in, int D_in, int H_in, int W_in, int C_out, int Kd, int Kh, int Kw, 
                  int sd, int sh, int sw, int pd, int ph, int pw, int dd, int dh, int dw, int groups,
                  int D_out, int H_out, int W_out, int smem) {
    int blocks = N * H_out * W_out;
    int threads = C_out;
    fused_conv_min_softmax_kernel<<<blocks, threads, smem>>>(
        x.data_ptr<float>(), weight.data_ptr<float>(), (bias.defined() ? bias.data_ptr<float>() : nullptr),
        out.data_ptr<float>(), N, C_in, D_in, H_in, W_in, C_out, Kd, Kh, Kw, sd, sh, sw, pd, ph, pw, dd, dh, dw, groups, 
        D_out, H_out, W_out);
}
"""

fused_ext = load_inline(name='fused_op', cpp_sources=cpp_source, cuda_sources=[cuda_source, cuda_wrapper], 
                        extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True)

def functional_model(x, *, conv_weight, conv_bias, conv_stride, conv_padding, 
                     conv_dilation, conv_groups, dim):
    x = x.contiguous()
    N, C_in, D_in, H_in, W_in = x.shape
    C_out, _, Kd, Kh, Kw = conv_weight.shape
    
    sd, sh, sw = (conv_stride, conv_stride, conv_stride) if isinstance(conv_stride, int) else conv_stride
    pd, ph, pw = (conv_padding, conv_padding, conv_padding) if isinstance(conv_padding, int) else conv_padding
    dd, dh, dw = (conv_dilation, conv_dilation, conv_dilation) if isinstance(conv_dilation, int) else conv_dilation
    
    D_out = (D_in + 2 * pd - dd * (Kd - 1) - 1) // sd + 1
    H_out = (H_in + 2 * ph - dh * (Kh - 1) - 1) // sh + 1
    W_out = (W_in + 2 * pw - dw * (Kw - 1) - 1) // sw + 1
    
    out = torch.empty((N, C_out, H_out, W_out), device=x.device)
    fused_ext.fused_op(x, conv_weight, conv_bias if conv_bias is not None else torch.tensor([]), out,
                       N, C_in, D_in, H_in, W_in, C_out, Kd, Kh, Kw, sd, sh, sw, pd, ph, pw, dd, dh, dw, conv_groups, 
                       D_out, H_out, W_out, (C_out * 2) * 4)
    return out
