# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_032753/code_7.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'max_pool1_kernel_size', 'max_pool1_stride', 'max_pool1_padding', 'max_pool1_dilation', 'max_pool1_ceil_mode', 'max_pool1_return_indices', 'max_pool2_kernel_size', 'max_pool2_stride', 'max_pool2_padding', 'max_pool2_dilation', 'max_pool2_ceil_mode', 'max_pool2_return_indices']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a 3D transposed convolution, followed by two max pooling layers and a sum operation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.max_pool1 = nn.MaxPool3d(kernel_size=2)
        self.max_pool2 = nn.MaxPool3d(kernel_size=3)

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
    # State for conv_transpose (nn.ConvTranspose3d)
    if 'conv_transpose_weight' in flat_state:
        state_kwargs['conv_transpose_weight'] = flat_state['conv_transpose_weight']
    else:
        state_kwargs['conv_transpose_weight'] = getattr(model.conv_transpose, 'weight', None)
    if 'conv_transpose_bias' in flat_state:
        state_kwargs['conv_transpose_bias'] = flat_state['conv_transpose_bias']
    else:
        state_kwargs['conv_transpose_bias'] = getattr(model.conv_transpose, 'bias', None)
    state_kwargs['conv_transpose_stride'] = model.conv_transpose.stride
    state_kwargs['conv_transpose_padding'] = model.conv_transpose.padding
    state_kwargs['conv_transpose_output_padding'] = model.conv_transpose.output_padding
    state_kwargs['conv_transpose_groups'] = model.conv_transpose.groups
    state_kwargs['conv_transpose_dilation'] = model.conv_transpose.dilation
    # State for max_pool1 (nn.MaxPool3d)
    state_kwargs['max_pool1_kernel_size'] = model.max_pool1.kernel_size
    state_kwargs['max_pool1_stride'] = model.max_pool1.stride
    state_kwargs['max_pool1_padding'] = model.max_pool1.padding
    state_kwargs['max_pool1_dilation'] = model.max_pool1.dilation
    state_kwargs['max_pool1_ceil_mode'] = model.max_pool1.ceil_mode
    state_kwargs['max_pool1_return_indices'] = model.max_pool1.return_indices
    # State for max_pool2 (nn.MaxPool3d)
    state_kwargs['max_pool2_kernel_size'] = model.max_pool2.kernel_size
    state_kwargs['max_pool2_stride'] = model.max_pool2.stride
    state_kwargs['max_pool2_padding'] = model.max_pool2.padding
    state_kwargs['max_pool2_dilation'] = model.max_pool2.dilation
    state_kwargs['max_pool2_ceil_mode'] = model.max_pool2.ceil_mode
    state_kwargs['max_pool2_return_indices'] = model.max_pool2.return_indices
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
# CUDA source – kernels
# -------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <algorithm>

// Fused kernel performing Transposed Conv3d + Bias (Implicit Add) + MaxPool + Sum
// Due to the requirement of not using built-in conv/matmul, we perform the 
// accumulation manually. To handle memory constraints and correctness, we 
// explicitly implement the stages.

__global__ void conv_transpose_kernel(
    const float* __restrict__ input, const float* __restrict__ weight,
    float* __restrict__ output,
    const int N, const int C_in, const int C_out, const int D_in, const int H_in, const int W_in,
    const int K, const int S, const int P, const int D,
    const int D_out, const int H_out, const int W_out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C_out * D_out * H_out * W_out) return;

    int tmp = idx;
    int w_out = tmp % W_out; tmp /= W_out;
    int h_out = tmp % H_out; tmp /= H_out;
    int d_out = tmp % D_out; tmp /= D_out;
    int c_out = tmp % C_out; tmp /= C_out;
    int n     = tmp;

    float val = 0.0f;
    for (int c_in = 0; c_in < C_in; ++c_in) {
        for (int kd = 0; kd < K; ++kd) {
            int d_in = (d_out + P - kd * D) / S;
            if (d_in * S != d_out + P - kd * D || d_in < 0 || d_in >= D_in) continue;
            for (int kh = 0; kh < K; ++kh) {
                int h_in = (h_out + P - kh * D) / S;
                if (h_in * S != h_out + P - kh * D || h_in < 0 || h_in >= H_in) continue;
                for (int kw = 0; kw < K; ++kw) {
                    int w_in = (w_out + P - kw * D) / S;
                    if (w_in * S != w_out + P - kw * D || w_in < 0 || w_in >= W_in) continue;
                    
                    int in_idx = (((n * C_in + c_in) * D_in + d_in) * H_in + h_in) * W_in + w_in;
                    int w_idx = ((c_in * C_out + c_out) * K + kd) * K * K + kh * K + kw;
                    val += input[in_idx] * weight[w_idx];
                }
            }
        }
    }
    output[idx] = val;
}

__global__ void max_pool_kernel(const float* input, float* output, 
    int N, int C, int D_in, int H_in, int W_in, int K, int S, int P,
    int D_out, int H_out, int W_out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C * D_out * H_out * W_out) return;
    int tmp = idx;
    int w_o = tmp % W_out; tmp /= W_out;
    int h_o = tmp % H_out; tmp /= H_out;
    int d_o = tmp % D_out; tmp /= D_out;
    int c   = tmp % C; tmp /= C;
    int n   = tmp;

    float max_val = -1e20f;
    for(int kd=0; kd<K; ++kd) for(int kh=0; kh<K; ++kh) for(int kw=0; kw<K; ++kw) {
        int di = d_o * S - P + kd, hi = h_o * S - P + kh, wi = w_o * S - P + kw;
        if(di >= 0 && di < D_in && hi >= 0 && hi < H_in && wi >= 0 && wi < W_in) {
            max_val = fmaxf(max_val, input[(((n * C + c) * D_in + di) * H_in + hi) * W_in + wi]);
        }
    }
    output[idx] = max_val;
}

void launch_conv(torch::Tensor in, torch::Tensor w, torch::Tensor out, int C_o, int K, int S, int P, int D, int Do, int Ho, int Wo) {
    int N = in.size(0), Ci = in.size(1), Di = in.size(2), Hi = in.size(3), Wi = in.size(4);
    int total = N * C_o * Do * Ho * Wo;
    conv_transpose_kernel<<<(total + 255)/256, 256>>>(in.data_ptr<float>(), w.data_ptr<float>(), out.data_ptr<float>(), N, Ci, C_o, Di, Hi, Wi, K, S, P, D, Do, Ho, Wo);
}

void launch_pool(torch::Tensor in, torch::Tensor out, int K, int S, int P) {
    int N=in.size(0), C=in.size(1), Di=in.size(2), Hi=in.size(3), Wi=in.size(4);
    int Do=out.size(2), Ho=out.size(3), Wo=out.size(4);
    max_pool_kernel<<<(N*C*Do*Ho*Wo + 255)/256, 256>>>(in.data_ptr<float>(), out.data_ptr<float>(), N, C, Di, Hi, Wi, K, S, P, Do, Ho, Wo);
}
"""

cpp_source = "void launch_conv(torch::Tensor i, torch::Tensor w, torch::Tensor o, int C, int K, int S, int P, int D, int Do, int Ho, int Wo);\n" \
             "void launch_pool(torch::Tensor i, torch::Tensor o, int K, int S, int P);\n" \
             "PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def(\"conv\", &launch_conv); m.def(\"pool\", &launch_pool); }"

fused_ext = load_inline(name='fused_custom', cpp_sources=cpp_source, cuda_sources=cuda_source, extra_cuda_cflags=['-O3'])

def functional_model(x, *, conv_transpose_weight, conv_transpose_bias, conv_transpose_stride, conv_transpose_padding, conv_transpose_output_padding, conv_transpose_groups, conv_transpose_dilation, max_pool1_kernel_size, max_pool1_stride, max_pool1_padding, max_pool1_dilation, max_pool1_ceil_mode, max_pool1_return_indices, max_pool2_kernel_size, max_pool2_stride, max_pool2_padding, max_pool2_dilation, max_pool2_ceil_mode, max_pool2_return_indices):
    N, Ci, Di, Hi, Wi = x.shape
    Co, K = conv_transpose_weight.shape[1], conv_transpose_weight.shape[2]
    Do = (Di - 1) * conv_transpose_stride - 2 * conv_transpose_padding + conv_transpose_dilation * (K - 1) + conv_transpose_output_padding + 1
    out = torch.empty(N, Co, Do, Do, Do, device=x.device)
    fused_ext.conv(x, conv_transpose_weight, out, Co, K, conv_transpose_stride, conv_transpose_padding, conv_transpose_dilation, Do, Do, Do)
    if conv_transpose_bias is not None: out += conv_transpose_bias.view(1, -1, 1, 1, 1)
    for k, s, p in [(max_pool1_kernel_size, max_pool1_stride, max_pool1_padding), (max_pool2_kernel_size, max_pool2_stride, max_pool2_padding)]:
        Do = (Do + 2 * p - k) // s + 1
        out2 = torch.empty(N, Co, Do, Do, Do, device=x.device)
        fused_ext.pool(out, out2, k, s, p)
        out = out2
    return out.sum(dim=1, keepdim=True)
