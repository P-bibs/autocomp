# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_141042/code_1.py
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

# Optimization: Fusing Conv2d (simplified sliding window implementation), 
# subtractions, and Mish activation into a single GPU pass to minimize Global Memory access.

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_conv_mish_kernel(
    const float* __restrict__ x, const float* __restrict__ weight, const float* __restrict__ bias,
    float* __restrict__ out, int N, int C_in, int C_out, int H, int W, int K,
    int pad_h, int pad_w, int stride_h, int stride_w, int dilation_h, int dilation_w,
    float sub1, float sub2) {
    
    int n = blockIdx.z;
    int co = blockIdx.y;
    int oh = blockIdx.x * blockDim.x + threadIdx.x;
    int ow = threadIdx.y;
    
    if (oh >= H || ow >= W) return;

    float acc = bias[co];
    for (int ci = 0; ci < C_in; ++ci) {
        for (int kh = 0; kh < K; ++kh) {
            for (int kw = 0; kw < K; ++kw) {
                int ih = oh * stride_h - pad_h + kh * dilation_h;
                int iw = ow * stride_w - pad_w + kw * dilation_w;
                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                    acc += x[((n * C_in + ci) * H + ih) * W + iw] * 
                           weight[((co * C_in + ci) * K + kh) * K + kw];
                }
            }
        }
    }
    
    // Fused operations: (x - sub1 - sub2) -> Mish(x) = x * tanh(ln(1 + exp(x)))
    float val = acc - sub1 - sub2;
    float exp_val = expf(val);
    float softplus_val = logf(1.0f + exp_val);
    out[((n * C_out + co) * H + oh) * W + ow] = val * tanhf(softplus_val);
}

void fused_op_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, torch::Tensor out, 
                      int pad_h, int pad_w, int stride_h, int stride_w, int dilation_h, int dilation_w,
                      float s1, float s2) {
    int N = x.size(0), C_in = x.size(1), H_in = x.size(2), W_in = x.size(3);
    int C_out = weight.size(0), K = weight.size(2);
    
    // Calculate output dimensions
    int H_out = (H_in + 2 * pad_h - dilation_h * (K - 1) - 1) / stride_h + 1;
    int W_out = (W_in + 2 * pad_w - dilation_w * (K - 1) - 1) / stride_w + 1;
    
    dim3 blocks((H_out + 15) / 16, C_out, N);
    dim3 threads(16, 16);
    fused_conv_mish_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        out.data_ptr<float>(), N, C_in, C_out, H_out, W_out, K, 
        pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, s1, s2);
}
"""

cpp_source = r"""
void fused_op_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, torch::Tensor out,
                      int pad_h, int pad_w, int stride_h, int stride_w, int dilation_h, int dilation_w,
                      float s1, float s2);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused Conv + Subtract + Mish operation");
}
"""

fused_ext = load_inline(
    name='fused_op', cpp_sources=cpp_source, cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True
)

def functional_model(x, *, conv_weight, conv_bias, conv_stride, conv_padding, 
                     conv_dilation, conv_groups, subtract_value_1, subtract_value_2):
    # Support for general conv parameters
    if isinstance(conv_stride, int):
        stride_h = stride_w = conv_stride
    else:
        stride_h, stride_w = conv_stride
        
    if isinstance(conv_padding, int):
        pad_h = pad_w = conv_padding
    else:
        pad_h, pad_w = conv_padding
        
    if isinstance(conv_dilation, int):
        dilation_h = dilation_w = conv_dilation
    else:
        dilation_h, dilation_w = conv_dilation

    # Calculate output dimensions
    H_in, W_in = x.size(2), x.size(3)
    K = conv_weight.size(2)
    H_out = (H_in + 2 * pad_h - dilation_h * (K - 1) - 1) // stride_h + 1
    W_out = (W_in + 2 * pad_w - dilation_w * (K - 1) - 1) // stride_w + 1
    
    out = torch.empty((x.size(0), conv_weight.size(0), H_out, W_out), device=x.device, dtype=x.dtype)
    fused_ext.fused_op(x, conv_weight, conv_bias, out, 
                       pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
                       subtract_value_1, subtract_value_2)
    return out

# Batch size and input dimensions for testing
batch_size = 128
in_channels = 8
out_channels = 64
height, width = 256, 256
kernel_size = 3
subtract_value_1 = 0.5
subtract_value_2 = 0.2

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2]

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]
