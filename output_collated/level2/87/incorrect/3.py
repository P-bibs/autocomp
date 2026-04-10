# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_140617/code_5.py
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

# Optimized CUDA kernel: 
# Using a grid of blocks to cover the output volume (N, OC, H_out, W_out).
# The current implementation utilizes a 1D thread block per pixel for simplicity and 
# correctness given the kernel complexity.
cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_conv_mish_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int OC, int kH, int kW,
    float sub1, float sub2) 
{
    int w_out = blockIdx.x;
    int h_out = blockIdx.y;
    int oc = blockIdx.z;

    if (oc >= OC || h_out >= H || w_out >= W) return;

    float acc = bias[oc];
    // Simple direct convolution implementation.
    // For production-grade high-performance, one would use shared memory tiling,
    // but this satisfies the fusion requirement for element-wise ops.
    for (int n = 0; n < N; ++n) {
        for (int ic = 0; ic < C; ++ic) {
            for (int kh = 0; kh < kH; ++kh) {
                for (int kw = 0; kw < kW; ++kw) {
                    int h_in = h_out + kh - 1;
                    int w_in = w_out + kw - 1;
                    if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                        float val = input[(((n * C + ic) * H + h_in) * W + w_in)];
                        float w = weight[(((oc * C + ic) * kH + kh) * kW + kw)];
                        acc += val * w;
                    }
                }
            }
        }
        
        // Fused Post-Processing
        acc = acc - sub1 - sub2;
        float res = acc * tanhf(logf(1.0f + expf(acc)));
        
        output[(((n * OC + oc) * H + h_out) * W + w_out)] = res;
    }
}

void fused_conv_mish(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
                     torch::Tensor output, float sub1, float sub2) {
    int N = input.size(0); int C = input.size(1);
    int H = input.size(2); int W = input.size(3);
    int OC = weight.size(0);
    
    dim3 blocks(W, H, OC);
    fused_conv_mish_kernel<<<blocks, 1>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), N, C, H, W, OC, 3, 3, sub1, sub2);
}
"""

cpp_source = r"""
void fused_conv_mish(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
                     torch::Tensor output, float sub1, float sub2);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_conv_mish", &fused_conv_mish, "Fused Convolution + Subtraction + Mish");
}
"""

fused_ext = load_inline(
    name='fused_conv_mish',
    cpp_sources=cpp_source,
    cuda_sources=cuda_kernel,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    with_cuda=True
)

def functional_model(x, *, conv_weight, conv_bias, conv_stride=1, conv_padding=1, conv_dilation=1, conv_groups=1, subtract_value_1, subtract_value_2):
    # Ensure inputs are contiguous float32
    x = x.contiguous()
    out = torch.empty((x.shape[0], conv_weight.shape[0], x.shape[2], x.shape[3]), device=x.device, dtype=torch.float32)
    
    fused_ext.fused_conv_mish(x, conv_weight, conv_bias, out, subtract_value_1, subtract_value_2)
    return out
