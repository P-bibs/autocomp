# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_141921/code_11.py
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

# The implementation below uses a manually fused CUDA kernel that performs
# the 2D convolution (sliding window), subtracts two constant values, 
# and computes the Mish activation function (x * tanh(softplus(x)))
# in a single pass to maximize cache locality and reduce DRAM pressure.

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__device__ __forceinline__ float softplus(float x) {
    // Standard approximation for softplus to avoid overflow
    return (x > 20.0f) ? x : logf(1.0f + expf(x));
}

__global__ void fused_conv_mish_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch, int in_c, int in_h, int in_w,
    int out_c, int k_size, int out_h, int out_w, int pad,
    float sub1, float sub2) {

    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= batch * out_c * out_h * out_w) return;

    int tmp = out_idx;
    int w_out = tmp % out_w; tmp /= out_w;
    int h_out = tmp % out_h; tmp /= out_h;
    int c_out = tmp % out_c; tmp /= out_c;
    int b = tmp;

    float acc = bias[c_out];

    for (int ic = 0; ic < in_c; ++ic) {
        for (int kh = 0; kh < k_size; ++kh) {
            for (int kw = 0; kw < k_size; ++kw) {
                int h_in = h_out + kh - pad;
                int w_in = w_out + kw - pad;
                
                if (h_in >= 0 && h_in < in_h && w_in >= 0 && w_in < in_w) {
                    float val = input[((b * in_c + ic) * in_h + h_in) * in_w + w_in];
                    float w = weight[(((c_out * in_c + ic) * k_size + kh) * k_size + kw)];
                    acc += val * w;
                }
            }
        }
    }

    float res = acc - sub1 - sub2;
    output[out_idx] = res * tanhf(softplus(res));
}

void fused_op(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
              torch::Tensor output, float sub1, float sub2, int padding) {
    int batch = input.size(0);
    int in_c = input.size(1);
    int in_h = input.size(2);
    int in_w = input.size(3);
    int out_c = weight.size(0);
    int k_size = weight.size(2);
    int out_h = input.size(2);
    int out_w = input.size(3);

    int total_elements = batch * out_c * out_h * out_w;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    fused_conv_mish_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), batch, in_c, in_h, in_w, out_c, k_size, out_h, out_w, padding, sub1, sub2);
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_op(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
              torch::Tensor output, float sub1, float sub2, int padding);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op, "Fused Conv2d + Subtracts + Mish");
}
"""

fused_ext = load_inline(
    name='fused_op', cpp_sources=cpp_source, cuda_sources=cuda_source,
    extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True)

def functional_model(x, *, conv_weight, conv_bias, conv_stride, conv_padding, conv_dilation, 
                     conv_groups, subtract_value_1, subtract_value_2):
    # Ensure inputs are contiguous float tensors
    x = x.contiguous()
    conv_weight = conv_weight.contiguous()
    conv_bias = conv_bias.contiguous()
    
    # Kernel assumes padding as 1 (defined by kernel_size 3)
    padding = conv_padding[0] if isinstance(conv_padding, tuple) else conv_padding
    
    output = torch.empty((x.size(0), conv_weight.size(0), x.size(2), x.size(3)), 
                         device=x.device, dtype=x.dtype)
    
    fused_ext.fused_op(x, conv_weight, conv_bias, output, 
                       float(subtract_value_1), float(subtract_value_2), int(padding))
    return output
