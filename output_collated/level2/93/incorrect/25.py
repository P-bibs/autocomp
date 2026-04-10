# Original path: /home/paulbib/Development/autocomp/autocomp/backend/kernelbench/tmp_files/kb_eval_20260409_153448/code_12.py
import torch
import torch.nn as nn
INIT_PARAM_NAMES = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'add_value', 'multiply_value']
FORWARD_ARG_NAMES = ['x']
FORWARD_FREE_VARS = []
REQUIRED_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias', 'conv_transpose_stride', 'conv_transpose_padding', 'conv_transpose_output_padding', 'conv_transpose_groups', 'conv_transpose_dilation', 'add_value', 'multiply_value']
REQUIRED_FLAT_STATE_NAMES = ['conv_transpose_weight', 'conv_transpose_bias']


class ModelNew(nn.Module):
    """
    ModelNew that performs a transposed convolution, adds a value, takes the minimum, applies GELU, and multiplies by a value.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, add_value, multiply_value):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.add_value = add_value
        self.multiply_value = multiply_value

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
    # State for conv_transpose (nn.ConvTranspose2d)
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
    if 'add_value' in flat_state:
        state_kwargs['add_value'] = flat_state['add_value']
    else:
        state_kwargs['add_value'] = getattr(model, 'add_value')
    if 'multiply_value' in flat_state:
        state_kwargs['multiply_value'] = flat_state['multiply_value']
    else:
        state_kwargs['multiply_value'] = getattr(model, 'multiply_value')
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

cuda_kernel = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__device__ __forceinline__ float fast_gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

__global__ void fused_conv_tr_kernel(
    const float* __restrict__ input, const float* __restrict__ weight, const float* __restrict__ bias,
    float* __restrict__ output, float add_val, float mul_val,
    int B, int Ci, int Co, int Hi, int Wi, int Hw, int Ww,
    int stride, int pad, int Ho, int Wo) {

    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * Co * Ho * Wo;
    if (gid >= total_elements) return;

    int idx = gid;
    int w_out = idx % Wo; idx /= Wo;
    int h_out = idx % Ho; idx /= Ho;
    int c_out = idx % Co; idx /= Co;
    int b = idx;

    // Direct Transposed Convolution (Scatter-Add) Implementation
    float acc = bias[c_out];
    for (int ci = 0; ci < Ci; ++ci) {
        for (int kh = 0; kh < Hw; ++kh) {
            for (int kw = 0; kw < Ww; ++kw) {
                int h_in = (h_out + pad - kh);
                int w_in = (w_out + pad - kw);
                if (h_in >= 0 && h_in < Hi * stride && w_in >= 0 && w_in < Wi * stride &&
                    h_in % stride == 0 && w_in % stride == 0) {
                    acc += input[((b * Ci + ci) * Hi + (h_in / stride)) * Wi + (w_in / stride)] * 
                           weight[(((ci * Co + c_out) * Hw + kh) * Ww + kw)];
                }
            }
        }
    }

    acc += add_val;
    acc = fminf(acc, 0.0f);
    output[gid] = fast_gelu(acc) * mul_val;
}

void fused_op_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
                      torch::Tensor output, float add_val, float mul_val, int stride, int pad) {
    int B = input.size(0); int Ci = input.size(1); int Hi = input.size(2); int Wi = input.size(3);
    int Co = weight.size(1); int Hw = weight.size(2); int Ww = weight.size(3);
    int Ho = (Hi - 1) * stride - 2 * pad + Hw;
    int Wo = (Wi - 1) * stride - 2 * pad + Ww;
    
    int num_elements = B * Co * Ho * Wo;
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;
    
    fused_conv_tr_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), add_val, mul_val, B, Ci, Co, Hi, Wi, Hw, Ww, stride, pad, Ho, Wo
    );
}
"""

cpp_source = r"""
#include <torch/extension.h>
void fused_op_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
                      torch::Tensor output, float add_val, float mul_val, int stride, int pad);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_op", &fused_op_forward, "Fused Transposed Conv + Add + Min + GELU + Mul");
}
"""

fused_ext = load_inline(name='fused_op', cpp_sources=cpp_source, cuda_sources=cuda_kernel, 
                       extra_cuda_cflags=['-O3', '--use_fast_math'], with_cuda=True)

def functional_model(x, *, conv_transpose_weight, conv_transpose_bias, conv_transpose_stride, 
                     conv_transpose_padding, conv_transpose_output_padding, conv_transpose_groups, 
                     conv_transpose_dilation, add_value, multiply_value):
    # Calculate output dimensions
    B, Ci, Hi, Wi = x.shape
    Co, _, Hw, Ww = conv_transpose_weight.shape
    Ho = (Hi - 1) * conv_transpose_stride - 2 * conv_transpose_padding + Hw + conv_transpose_output_padding
    Wo = (Wi - 1) * conv_transpose_stride - 2 * conv_transpose_padding + Ww + conv_transpose_output_padding
    
    out = torch.empty((B, Co, Ho, Wo), device=x.device, dtype=x.dtype)
    fused_ext.fused_op(x, conv_transpose_weight, conv_transpose_bias, out, 
                       float(add_value), float(multiply_value), 
                       conv_transpose_stride, conv_transpose_padding)
    return out
