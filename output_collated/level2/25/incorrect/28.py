# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_083856/code_15.py
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
#  Inline CUDA source – fused convolution + min + two tanh
# -------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <algorithm>

// Fused kernel: Computes conv2d, minimizes over channels, and applies tanh twice
__global__ void fused_conv_min_tanh_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int N, const int C_in, const int H, const int W,
    const int C_out, const int K,
    const int stride, const int pad, const int dilation,
    const int out_h, const int out_w)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * out_h * out_w) return;

    // Decode batch, height, width from spatial index
    const int n = idx / (out_h * out_w);
    const int out_rem = idx % (out_h * out_w);
    const int h = out_rem / out_w;
    const int w = out_rem % out_w;

    // Accumulate results for each output channel in registers
    float vals[64]; 
    if (bias) {
        for (int co = 0; co < C_out; ++co) vals[co] = bias[co];
    } else {
        for (int co = 0; co < C_out; ++co) vals[co] = 0.0f;
    }

    // Naive sliding window convolution
    for (int ci = 0; ci < C_in; ++ci) {
        for (int kh = 0; kh < K; ++kh) {
            int in_row = h * stride + kh * dilation - pad;
            if (in_row < 0 || in_row >= H) continue;
            for (int kw = 0; kw < K; ++kw) {
                int in_col = w * stride + kw * dilation - pad;
                if (in_col < 0 || in_col >= W) continue;

                float inp = input[((n * C_in + ci) * H + in_row) * W + in_col];
                
                for (int co = 0; co < C_out; ++co) {
                    // weight layout: (C_out, C_in, K, K)
                    int w_idx = ((co * C_in + ci) * K + kh) * K + kw;
                    vals[co] += weight[w_idx] * inp;
                }
            }
        }
    }

    // Reduction and activation
    float min_val = vals[0];
    for (int co = 1; co < C_out; ++co) {
        if (vals[co] < min_val) min_val = vals[co];
    }

    float t1 = tanhf(min_val);
    output[idx] = tanhf(t1);
}

void fused_op_forward(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output,
    int N, int C_in, int H, int W, int C_out, int K,
    int stride, int pad, int dilation, int out_h, int out_w)
{
    const int threads = 256;
    const int total_elements = N * out_h * out_w;
    const int blocks = (total_elements + threads - 1) / threads;

    fused_conv_min_tanh_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        N, C_in, H, W, C_out, K, stride, pad, dilation, out_h, out_w
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_op_forward(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int, int, int, int, int, int, int, int, int, int, int);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused kernel");
}
"""

fused_ext = load_inline(
    name='fused_op', cpp_sources=cpp_source, cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True
)

def functional_model(x, *, conv_weight, conv_bias, conv_stride, conv_padding, conv_dilation, conv_groups):
    x = x.contiguous().cuda()
    conv_weight = conv_weight.contiguous().cuda()
    if conv_bias is not None: conv_bias = conv_bias.contiguous().cuda()
    
    H, W = x.shape[2], x.shape[3]
    K = conv_weight.shape[2]
    out_h = (H + 2 * conv_padding - conv_dilation * (K - 1) - 1) // conv_stride + 1
    out_w = (W + 2 * conv_padding - conv_dilation * (K - 1) - 1) // conv_stride + 1
    
    output = torch.empty((x.shape[0], 1, out_h, out_w), device='cuda')
    
    fused_ext.fused_op(
        x, conv_weight, conv_bias, output,
        x.shape[0], x.shape[1], H, W, conv_weight.shape[0], K,
        conv_stride, conv_padding, conv_dilation, out_h, out_w
    )
    return output
