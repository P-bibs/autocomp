# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_091623/code_15.py
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

# -------------------------------------------------------------------------
# CUDA source – grouped convolution and fused min + double tanh
# -------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__global__ void conv_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int N, const int CI, const int IH, const int IW,
    const int CO, const int OH, const int OW,
    const int KH, const int KW,
    const int stride, const int padding, const int dilation, const int groups)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * CO * OH * OW;
    if (idx >= total) return;

    int tmp = idx;
    int ow = tmp % OW; tmp /= OW;
    int oh = tmp % OH; tmp /= OH;
    int co = tmp % CO; tmp /= CO;
    int n  = tmp;

    int CI_per_group = CI / groups;
    int CO_per_group = CO / groups;
    int group_idx = co / CO_per_group;
    int ci_base = group_idx * CI_per_group;

    float sum = 0.0f;
    for (int i = 0; i < CI_per_group; ++i) {
        int ci = ci_base + i;
        for (int kh = 0; kh < KH; ++kh) {
            int ih = oh * stride - padding + kh * dilation;
            if (ih < 0 || ih >= IH) continue;
            for (int kw = 0; kw < KW; ++kw) {
                int iw = ow * stride - padding + kw * dilation;
                if (iw < 0 || iw >= IW) continue;
                
                float w = weight[((co * CI_per_group + i) * KH + kh) * KW + kw];
                float v = input[((n * CI + ci) * IH + ih) * IW + iw];
                sum += w * v;
            }
        }
    }
    if (bias) sum += bias[co];
    output[idx] = sum;
}

__global__ void min_tanh_kernel(
    const float* __restrict__ in,
    float* __restrict__ out,
    const int N, const int CO, const int OH, const int OW)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * OH * OW;
    if (idx >= total) return;

    int n = idx / (OH * OW);
    int spatial = idx % (OH * OW);
    
    float min_val = 1e38f;
    for (int co = 0; co < CO; ++co) {
        float val = in[((n * CO + co) * OH + OW) + spatial];
        if (val < min_val) min_val = val;
    }
    
    float t1 = tanhf(min_val);
    out[idx] = tanhf(t1);
}

void launch_conv(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
                 torch::Tensor output, int stride, int padding, int dilation, int groups) {
    const int N = input.size(0); const int CI = input.size(1);
    const int IH = input.size(2); const int IW = input.size(3);
    const int CO = weight.size(0); const int KH = weight.size(2); const int KW = weight.size(3);
    const int OH = output.size(2); const int OW = output.size(3);
    
    int total = N * CO * OH * OW;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    
    conv_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), N, CI, IH, IW, CO, OH, OW, KH, KW, 
        stride, padding, dilation, groups);
}

void launch_min_tanh(torch::Tensor in, torch::Tensor out) {
    const int N = in.size(0); const int CO = in.size(1); 
    const int OH = in.size(2); const int OW = in.size(3);
    
    int total = N * OH * OW;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    
    min_tanh_kernel<<<blocks, threads>>>(in.data_ptr<float>(), out.data_ptr<float>(), N, CO, OH, OW);
}
"""

cpp_source = r"""
void launch_conv(at::Tensor, at::Tensor, at::Tensor, at::Tensor, int, int, int, int);
void launch_min_tanh(at::Tensor, at::Tensor);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv", &launch_conv);
    m.def("min_tanh", &launch_min_tanh);
}
"""

conv_ext = load_inline(name='conv_ext', cpp_sources=cpp_source, cuda_sources=cuda_source, 
                       extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True)

def functional_model(x, *, conv_weight, conv_bias, conv_stride, conv_padding, conv_dilation, conv_groups):
    device = x.device
    stride = conv_stride[0] if isinstance(conv_stride, (tuple, list)) else conv_stride
    padding = conv_padding[0] if isinstance(conv_padding, (tuple, list)) else conv_padding
    dilation = conv_dilation[0] if isinstance(conv_dilation, (tuple, list)) else conv_dilation
    
    N, CI, IH, IW = x.shape
    CO, _, KH, KW = conv_weight.shape
    OH = (IH + 2 * padding - dilation * (KH - 1) - 1) // stride + 1
    OW = (IW + 2 * padding - dilation * (KW - 1) - 1) // stride + 1
    
    conv_out = torch.empty(N, CO, OH, OW, device=device)
    conv_ext.conv(x, conv_weight, conv_bias, conv_out, stride, padding, dilation, conv_groups)
    
    out = torch.empty(N, 1, OH, OW, device=device)
    conv_ext.min_tanh(conv_out, out)
    return out
