# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_141921/code_15.py
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

# ----------------------------------------------------------------------
# 1. CUDA source – The fused convolution + subtract + mish kernel
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_conv_sub_mish_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ out,
    const int N, const int C_in, const int C_out,
    const int H_in, const int W_in,
    const int K_h, const int K_w,
    const int stride_h, const int stride_w,
    const int pad_h, const int pad_w,
    const int groups,
    const float sub1,
    const float sub2,
    const int H_out,
    const int W_out,
    const bool has_bias)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_out = N * C_out * H_out * W_out;
    if (idx >= total_out) return;

    // Decoding linear index to multi-dimensional indices
    int ow = idx % W_out;
    int tmp = idx / W_out;
    int oh = tmp % H_out;
    tmp = tmp / H_out;
    int co = tmp % C_out;
    int n  = tmp / C_out;

    const int C_in_per_group = C_in / groups;
    const int g = co / (C_out / groups);
    const int ci_start = g * C_in_per_group;

    float sum = has_bias ? bias[co] : 0.0f;

    // Convolution Logic
    float sum_acc = 0.0f;
    #pragma unroll
    for (int kh = 0; kh < K_h; ++kh) {
        int i_h = oh * stride_h - pad_h + kh;
        if (i_h < 0 || i_h >= H_in) continue;
        #pragma unroll
        for (int kw = 0; kw < K_w; ++kw) {
            int i_w = ow * stride_w - pad_w + kw;
            if (i_w < 0 || i_w >= W_in) continue;
            
            for (int ci = 0; ci < C_in_per_group; ++ci) {
                int ci_global = ci_start + ci;
                float x_val = __ldg(&x[((n * C_in + ci_global) * H_in + i_h) * W_in + i_w]);
                float w_val = __ldg(&weight[(((co * C_in_per_group + ci) * K_h + kh) * K_w + kw)]);
                sum_acc += x_val * w_val;
            }
        }
    }
    
    sum += sum_acc;
    sum -= (sub1 + sub2);

    // Mish Activation: x * tanh(softplus(x))
    // softplus(x) = log(1 + exp(x))
    float sp = logf(1.0f + expf(sum));
    out[idx] = sum * tanhf(sp);
}

void fused_op_forward(
    torch::Tensor x, torch::Tensor weight, torch::Tensor bias,
    int stride, int padding, int groups, float sub1, float sub2,
    torch::Tensor out) 
{
    const int N = x.size(0);
    const int C_in = x.size(1);
    const int H_in = x.size(2);
    const int W_in = x.size(3);
    const int C_out = weight.size(0);
    const int K_h = weight.size(2);
    const int K_w = weight.size(3);
    const int H_out = (H_in + 2 * padding - K_h) / stride + 1;
    const int W_out = (W_in + 2 * padding - K_w) / stride + 1;
    
    const int total = N * C_out * H_out * W_out;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    fused_conv_sub_mish_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(), weight.data_ptr<float>(), 
        bias.size(0) > 0 ? bias.data_ptr<float>() : nullptr,
        out.data_ptr<float>(), N, C_in, C_out, H_in, W_in, 
        K_h, K_w, stride, stride, padding, padding, groups, 
        sub1, sub2, H_out, W_out, bias.size(0) > 0
    );
}
"""

cpp_source = r"""
void fused_op_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, 
                      int stride, int padding, int groups, float sub1, float sub2, 
                      torch::Tensor out);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused Conv/Sub/Mish");
}
"""

fused_ext = load_inline(
    name='fused_op', cpp_sources=cpp_source, cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True
)

def functional_model(x, *, conv_weight, conv_bias, conv_stride, conv_padding, 
                     conv_dilation, conv_groups, subtract_value_1, subtract_value_2):
    x = x.cuda().contiguous()
    w = conv_weight.cuda().contiguous()
    b = conv_bias.cuda().contiguous() if conv_bias is not None else torch.tensor([], device='cuda')
    
    N, C_in, H_in, W_in = x.shape
    C_out, _, K_h, K_w = w.shape
    H_out = (H_in + 2 * conv_padding - K_h) // conv_stride + 1
    W_out = (W_in + 2 * conv_padding - K_w) // conv_stride + 1
    
    out = torch.empty((N, C_out, H_out, W_out), device='cuda')
    
    fused_ext.fused_op(x, w, b, conv_stride, conv_padding, conv_groups, 
                       subtract_value_1, subtract_value_2, out)
    return out

batch_size, in_channels, out_channels, height, width, kernel_size = 128, 8, 64, 256, 256, 3
subtract_value_1, subtract_value_2 = 0.5, 0.2

def get_init_inputs(): return [in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2]
def get_inputs(): return [torch.rand(batch_size, in_channels, height, width)]
