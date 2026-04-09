# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_081239/code_1.py
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

# Optimization: Fusing Conv2d, Min-Reduction, and Tanh into a single kernel
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

__global__ void fused_conv_min_tanh_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ weight, 
    const float* __restrict__ bias,
    float* __restrict__ output, 
    int N, int C_in, int C_out, int H, int W, int K,
    int pad_h, int pad_w, int stride_h, int stride_w, int dilation_h, int dilation_w) {
    
    int batch_idx = blockIdx.z;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx >= N || h_out >= (H + 2 * pad_h - (dilation_h * (K - 1) + 1)) / stride_h + 1 || 
        w_out >= (W + 2 * pad_w - (dilation_w * (K - 1) + 1)) / stride_w + 1) return;
    
    float min_val = 1e30f;
    
    for (int cout = 0; cout < C_out; ++cout) {
        float acc = bias[cout];
        
        for (int cin = 0; cin < C_in; ++cin) {
            for (int ki = 0; ki < K; ++ki) {
                for (int kj = 0; kj < K; ++kj) {
                    int h_in = h_out * stride_h - pad_h + ki * dilation_h;
                    int w_in = w_out * stride_w - pad_w + kj * dilation_w;
                    
                    if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                        acc += input[((batch_idx * C_in + cin) * H + h_in) * W + w_in] * 
                               weight[(((cout * C_in + cin) * K + ki) * K + kj)];
                    }
                }
            }
        }
        
        if (acc < min_val) min_val = acc;
    }
    
    float t = tanhf(min_val);
    output[((batch_idx * (((H + 2 * pad_h - (dilation_h * (K - 1) + 1)) / stride_h + 1))) + h_out) * 
           ((W + 2 * pad_w - (dilation_w * (K - 1) + 1)) / stride_w + 1) + w_out] = tanhf(t);
}

void fused_op_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output,
                      int pad_h, int pad_w, int stride_h, int stride_w, int dilation_h, int dilation_w) {
    int N = input.size(0);
    int C_in = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    int C_out = weight.size(0);
    int K = weight.size(2);
    
    int out_h = (H + 2 * pad_h - (dilation_h * (K - 1) + 1)) / stride_h + 1;
    int out_w = (W + 2 * pad_w - (dilation_w * (K - 1) + 1)) / stride_w + 1;
    
    dim3 block(16, 16);
    dim3 grid(CEIL_DIV(out_w, block.x), CEIL_DIV(out_h, block.y), N);
    
    fused_conv_min_tanh_kernel<<<grid, block>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), N, C_in, C_out, H, W, K,
        pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w);
}
"""

cpp_source = r"""
#include <torch/extension.h>

void fused_op_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output,
                      int pad_h, int pad_w, int stride_h, int stride_w, int dilation_h, int dilation_w);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op_forward", &fused_op_forward, "Fused Conv-Min-Tanh forward");
}
"""

fused_ext = load_inline(
    name='fused_op',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, conv_weight, conv_bias, conv_stride, conv_padding, conv_dilation, conv_groups):
    # Output shape: [batch, 1, H, W]
    out_h = (x.size(2) + 2 * conv_padding[0] - (conv_dilation[0] * (conv_weight.size(2) - 1) + 1)) // conv_stride[0] + 1
    out_w = (x.size(3) + 2 * conv_padding[1] - (conv_dilation[1] * (conv_weight.size(3) - 1) + 1)) // conv_stride[1] + 1
    out = torch.empty(x.size(0), 1, out_h, out_w, device=x.device, dtype=x.dtype)
    
    fused_ext.fused_op_forward(
        x, conv_weight, conv_bias, out,
        conv_padding[0], conv_padding[1],
        conv_stride[0], conv_stride[1],
        conv_dilation[0], conv_dilation[1]
    )
    return out

# Placeholders for setup compliance
batch_size, in_channels, out_channels, height, width, kernel_size = 128, 16, 64, 256, 256, 3

def get_init_inputs(): 
    return [in_channels, out_channels, kernel_size]

def get_inputs(): 
    return [torch.rand(batch_size, in_channels, height, width, device='cuda')]
