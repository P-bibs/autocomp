# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_140617/code_6.py
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

# CUDA Kernel implementation
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

__global__ void fused_op_forward_kernel(
    const float* __restrict__ in,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ out,
    const int N, const int C_in, const int C_out,
    const int H_in, const int W_in,
    const int K,
    const int stride, const int pad,
    const float sub1, const float sub2,
    const int H_out, const int W_out) 
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C_out * H_out * W_out;
    if (tid >= total_elements) return;

    // Calculate indices
    int tmp = tid;
    int w_out = tmp % W_out; tmp /= W_out;
    int h_out = tmp % H_out; tmp /= H_out;
    int c_out = tmp % C_out; tmp /= C_out;
    int n     = tmp;

    float acc = (bias != nullptr) ? bias[c_out] : 0.0f;

    // Convolution Logic (Direct spatial approach)
    int h_start = h_out * stride - pad;
    int w_start = w_out * stride - pad;

    for (int c = 0; c < C_in; ++c) {
        const float* weight_ptr = weight + ((c_out * C_in + c) * K * K);
        const float* in_ptr = in + ((n * C_in + c) * H_in * W_in);
        
        for (int kh = 0; kh < K; ++kh) {
            int ih = h_start + kh;
            if (ih < 0 || ih >= H_in) continue;
            for (int kw = 0; kw < K; ++kw) {
                int iw = w_start + kw;
                if (iw >= 0 && iw < W_in) {
                    acc += in_ptr[ih * W_in + iw] * weight_ptr[kh * K + kw];
                }
            }
        }
    }

    // Fused operations: Subtracts then Mish
    float val = acc - sub1 - sub2;
    // Mish = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    out[tid] = val * tanhf(log1pf(expf(val)));
}

void fused_op_forward(
    torch::Tensor in, torch::Tensor weight, torch::Tensor bias, torch::Tensor out,
    int stride, int pad, float sub1, float sub2) 
{
    int N = in.size(0), C_out = weight.size(0);
    int H_out = out.size(2), W_out = out.size(3);
    int total = N * C_out * H_out * W_out;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    fused_op_forward_kernel<<<blocks, threads>>>(
        in.data_ptr<float>(), weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        out.data_ptr<float>(),
        N, in.size(1), C_out, in.size(2), in.size(3),
        weight.size(2), stride, pad, sub1, sub2, H_out, W_out
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_op_forward(torch::Tensor in, torch::Tensor weight, torch::Tensor bias, torch::Tensor out, int stride, int pad, float sub1, float sub2);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused Conv2d-Sub-Mish");
}
"""

fused_ext = load_inline(
    name='fused_op', cpp_sources=cpp_source, cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True
)

def functional_model(x, *, conv_weight, conv_bias, conv_stride, conv_padding, conv_dilation, conv_groups, subtract_value_1, subtract_value_2):
    out_h = (x.shape[2] + 2 * conv_padding - conv_weight.shape[2]) // conv_stride + 1
    out_w = (x.shape[3] + 2 * conv_padding - conv_weight.shape[3]) // conv_stride + 1
    out = torch.empty((x.shape[0], conv_weight.shape[0], out_h, out_w), device=x.device, dtype=x.dtype)
    fused_ext.fused_op(x, conv_weight, conv_bias, out, conv_stride, conv_padding, subtract_value_1, subtract_value_2)
    return out

batch_size, in_channels, out_channels, height, width, kernel_size = 128, 8, 64, 256, 256, 3
subtract_value_1, subtract_value_2 = 0.5, 0.2

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]
